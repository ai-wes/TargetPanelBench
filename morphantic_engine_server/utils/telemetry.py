import os
import json
import time
import threading
import torch
import networkx as nx
import logging
from datetime import datetime
from scripts.core.depricated.training_config import cfg
from scripts.core.utils.detailed_logger import get_logger, trace
from torch import nn
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import asyncio
import queue


_event_emitter = None

# Set up logger
logger = logging.getLogger(__name__)

# Lock to serialize file writes
state_lock = threading.Lock()
_run_id = None

# Async telemetry writer
_telemetry_executor = None
_telemetry_queue = None
_telemetry_thread = None




import tempfile

def atomic_json_write(data, filename, **kwargs):
    dir_name = os.path.dirname(filename)
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8') as tf:
        try:
            # Always set indent=2 unless overridden
            if 'indent' not in kwargs:
                kwargs['indent'] = 2
            json.dump(data, tf, ensure_ascii=False, **kwargs)
            tf.flush()
            os.fsync(tf.fileno())
            tempname = tf.name
        except Exception as e:
            tf.close()
            os.unlink(tf.name)
            raise
    os.replace(tempname, filename)
    
    
    
def init_async_telemetry():
    """Initialize async telemetry writer"""
    global _telemetry_executor, _telemetry_queue, _telemetry_thread
    
    if _telemetry_executor is None:
        _telemetry_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TelemetryWriter")
        _telemetry_queue = queue.Queue(maxsize=100)
        _telemetry_thread = threading.Thread(target=_async_telemetry_worker, daemon=True)
        _telemetry_thread.start()
        logger.info("[TELEMETRY] Async telemetry writer initialized")

def _async_telemetry_worker():
    """Background worker for async telemetry writing"""
    while True:
        try:
            # Get write task from queue
            task = _telemetry_queue.get(timeout=1.0)
            if task is None:  # Stop signal
                break
                
            filepath, data = task
            
            # Write data asynchronously using atomic write
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                atomic_json_write(data, filepath, indent=2)
            except Exception as e:
                logger.error(f"[TELEMETRY] Failed to write {filepath}: {e}")
                
            _telemetry_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"[TELEMETRY] Worker error: {e}")

def shutdown_async_telemetry():
    """Shutdown async telemetry writer"""
    global _telemetry_executor, _telemetry_queue, _telemetry_thread
    
    if _telemetry_queue:
        # Wait for queue to empty
        _telemetry_queue.join()
        # Send stop signal
        _telemetry_queue.put(None)
        
    if _telemetry_thread:
        _telemetry_thread.join(timeout=5.0)
        
    if _telemetry_executor:
        _telemetry_executor.shutdown(wait=True)
        
    logger.info("[TELEMETRY] Async telemetry writer shutdown")

def get_config_dict(config_obj):
    """Extract all configuration attributes from a config object"""
    config_dict = {}
    for attr_name in dir(config_obj):
        # Skip private attributes and methods
        if attr_name.startswith('_') or callable(getattr(config_obj, attr_name)):
            continue
        
        try:
            value = getattr(config_obj, attr_name)
            # Convert non-serializable types to strings
            if isinstance(value, (torch.device, type)):
                value = str(value)
            config_dict[attr_name] = value
        except Exception as e:
            config_dict[attr_name] = f"Error accessing {attr_name}: {str(e)}"
    
    return config_dict


# Global variables for visualization
_current_germinal_center = None


def generate_run_id() -> str:
    """Generate a consistent datetime-based run ID.
    
    Returns:
        str: Run ID in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_drug_run_id() -> str:
    """Generate a consistent datetime-based run ID.
    
    Returns:
        str: Run ID in format YYYYMMDD_HHMMSS
    """
    return f"drug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"




def get_or_create_run_id() -> str:
    """Get the current run_id or create a new one if none exists."""
    global _run_id
    if cfg.enable_drug_discovery_telemetry:
        if not _run_id or not _run_id.startswith("drug_"):
            _run_id = generate_drug_run_id()
    else:
        if _run_id is None:
            _run_id = generate_run_id()
    return _run_id


def set_run_id(run_id: str) -> None:
    """Set the global run_id.
    
    Args:
        run_id: The run_id to set
    """
    global _run_id
    _run_id = run_id


def write_archipelago_visualization_state(archipelago, generation, run_id=None):
    """Writes a comprehensive visualization state for the entire archipelago including all islands,
    their populations, Hall of Fame, and Causal Tapestry summary to a single JSON file per generation."""
    
    import json
    import os
    import time
    from collections import defaultdict
    
    # Use datetime-based run_id if none provided
    if run_id is None:
        run_id = get_or_create_run_id()
    
    islands_data = []
    archipelago_summary = {
        'total_islands': len(archipelago.islands),
        'total_population': 0,
        'total_genes': 0,
        'active_genes': 0,
        'quantum_genes': 0,
        'generation': generation,
        'timestamp': time.time()
    }
    
    # Collect data from all islands
    for island_idx, island in enumerate(archipelago.islands):
        # Get island-specific configuration (may have overrides from base config)
        island_config = get_config_dict(cfg).copy()
        
        # Add any island-specific configuration overrides
        if hasattr(island, 'config_overrides'):
            island_config.update(island.config_overrides)
        
        island_data = {
            'island_id': island_idx,
            'island_name': getattr(island, 'island_desc', f'Island_{island_idx}'),
            'island_type': getattr(island, 'island_type', 'unknown'),
            'device': str(getattr(island, 'device', 'unknown')),
            'generation': getattr(island, 'generation', generation),
            'current_stress': getattr(island, 'current_stress', 0.0),
            'population_size': 0,
            'cells': [],
            'hall_of_fame': [],
            'island_stats': {},
            'configuration': island_config  # Individual island configuration
        }
        
        # Collect population data
        if hasattr(island, 'population') and island.population:
            island_data['population_size'] = len(island.population)
            archipelago_summary['total_population'] += len(island.population)
            
            for cell_idx, (cell_id, cell) in enumerate(island.population.items()):
                # Determine cell type based on dominant active genes
                type_counts = defaultdict(int)
                cell_type = 'balanced'

                if hasattr(cell, 'genes'):
                    active_genes = [g for g in cell.genes if g.is_active]
                    if active_genes:
                        for gene in active_genes:
                            type_counts[gene.gene_type] += 1
                        
                        if type_counts:
                            dominant_type = max(type_counts, key=type_counts.get)
                            
                            type_mapping = {
                                'S': 'stem',
                                'V': 'biosensor',
                                'D': 'effector',
                                'J': 'controller',
                                'Q': 'quantum'
                            }
                            cell_type = type_mapping.get(dominant_type, 'balanced')

                cell_info = {
                    'cell_id': str(cell_id),
                    'index': cell_idx,
                    'fitness': cell.fitness_history[-1] if cell.fitness_history else 0.0,
                    'generation': getattr(cell, 'generation', generation),
                    'lineage': getattr(cell, 'lineage', []),
                    'type': cell_type,
                    'genes': [],
                    'architecture': None,
                    'connections': []
                }
                
                # Collect gene information
                if hasattr(cell, 'genes'):
                    for gene in cell.genes:
                        gene_info = {
                            'gene_id': str(getattr(gene, 'gene_id', str(id(gene)))),
                            'gene_type': str(getattr(gene, 'gene_type', 'V')),
                            'position': int(getattr(gene, 'position', 0)),
                            'is_active': bool(getattr(gene, 'is_active', False)),
                            'is_quantum': 'Quantum' in gene.__class__.__name__,
                            'depth': float(gene.compute_depth().item()) if hasattr(gene, 'compute_depth') else 1.0,
                            'activation': float(getattr(gene, 'activation_ema', 0.0)),
                            'variant_id': int(getattr(gene, 'variant_id', 0)),
                            'methylation': float(gene.methylation_state.mean().item()) if hasattr(gene, 'methylation_state') else 0.0
                        }
                        cell_info['genes'].append(gene_info)
                        
                        # Update archipelago-wide gene counts
                        archipelago_summary['total_genes'] += 1
                        if gene_info['is_active']:
                            archipelago_summary['active_genes'] += 1
                        if gene_info['is_quantum']:
                            archipelago_summary['quantum_genes'] += 1
                    
                    # Track gene connections
                    active_genes = [g for g in cell.genes if g.is_active]
                    for idx1, gene1 in enumerate(active_genes):
                        for idx2, gene2 in enumerate(active_genes[idx1+1:], idx1+1):
                            cell_info['connections'].append({
                                'source': str(gene1.gene_id),
                                'target': str(gene2.gene_id),
                                'strength': float(abs(idx1 - idx2) / len(active_genes)) if active_genes else 0.0
                            })
                
                # Add architecture information if available
                if hasattr(cell, 'architecture_modifier'):
                    arch = cell.architecture_modifier
                    try:
                        module_names = []
                        if hasattr(arch, 'dynamic_modules'):
                            module_names = list(arch.dynamic_modules.keys())
                        
                        connections = {}
                        if hasattr(arch, 'module_connections'):
                            for k, v in list(arch.module_connections.items()):
                                connections[str(k)] = list(v) if isinstance(v, (list, set, tuple)) else [str(v)]
                        
                        cell_info['architecture'] = {
                            'dna': str(getattr(arch, 'architecture_dna', 'N/A')),
                            'modules': module_names,
                            'connections': connections,
                            'modifications': len(getattr(arch, 'modification_history', []))
                        }
                    except Exception as arch_error:
                        logger.warning(f"Failed to serialize architecture info: {arch_error}")
                        cell_info['architecture'] = {
                            'dna': 'error',
                            'modules': [],
                            'connections': {},
                            'modifications': 0
                        }
                
                island_data['cells'].append(cell_info)
        
        # Collect Hall of Fame data
        if hasattr(island, 'hall_of_fame') and island.hall_of_fame:
            for idx, (fitness, champion_cell) in enumerate(island.hall_of_fame):
                hall_of_fame_entry = {
                    'rank': idx + 1,
                    'fitness': float(fitness),
                    'cell_id': str(champion_cell.cell_id),
                    'generation': getattr(champion_cell, 'generation', 0),
                    'lineage': getattr(champion_cell, 'lineage', []),
                    'island': island_data['island_name']
                }
                if hasattr(champion_cell, 'genes'):
                    hall_of_fame_entry['genes'] = [
                        {
                            'gene_id': str(getattr(gene, 'gene_id', str(id(gene)))),
                            'gene_type': str(getattr(gene, 'gene_type', 'V')),
                            'fitness': float(getattr(gene, 'fitness_contribution', 0.0)),
                            'is_active': bool(getattr(gene, 'is_active', False)),
                            'is_quantum': 'Quantum' in gene.__class__.__name__
                        }
                        for gene in champion_cell.genes
                    ]
                island_data['hall_of_fame'].append(hall_of_fame_entry)
        
        # Calculate island-specific statistics
        if island_data['cells']:
            fitnesses = [cell['fitness'] for cell in island_data['cells']]
            island_data['island_stats'] = {
                'mean_fitness': sum(fitnesses) / len(fitnesses),
                'max_fitness': max(fitnesses),
                'min_fitness': min(fitnesses),
                'fitness_std': np.std(fitnesses) if len(fitnesses) > 1 else 0.0,
                'active_cells': sum(1 for cell in island_data['cells'] if cell['fitness'] > 0),
                'cell_types': {
                    'stem': sum(1 for cell in island_data['cells'] if cell['type'] == 'stem'),
                    'biosensor': sum(1 for cell in island_data['cells'] if cell['type'] == 'biosensor'),
                    'effector': sum(1 for cell in island_data['cells'] if cell['type'] == 'effector'),
                    'controller': sum(1 for cell in island_data['cells'] if cell['type'] == 'controller'),
                    'quantum': sum(1 for cell in island_data['cells'] if cell['type'] == 'quantum'),
                    'balanced': sum(1 for cell in island_data['cells'] if cell['type'] == 'balanced')
                }
            }
        
        # Calculate comprehensive island-specific KPIs
        island_kpis = {}
        
        # Basic population KPIs
        if island_data['cells']:
            fitnesses = [cell['fitness'] for cell in island_data['cells']]
            active_genes = sum(1 for cell in island_data['cells'] for g in cell.get('genes', []) if g.get('is_active', False))
            quantum_genes = sum(1 for cell in island_data['cells'] for g in cell.get('genes', []) if g.get('is_quantum', False))
            total_genes = sum(len(cell.get('genes', [])) for cell in island_data['cells'])
            
            island_kpis.update({
                'population_size': len(island_data['cells']),
                'mean_fitness': sum(fitnesses) / len(fitnesses),
                'max_fitness': max(fitnesses),
                'min_fitness': min(fitnesses),
                'fitness_std': np.std(fitnesses) if len(fitnesses) > 1 else 0.0,
                'fitness_range': max(fitnesses) - min(fitnesses),
                'total_genes': total_genes,
                'active_genes': active_genes,
                'quantum_genes': quantum_genes,
                'gene_activation_rate': active_genes / total_genes if total_genes > 0 else 0.0,
                'quantum_gene_ratio': quantum_genes / total_genes if total_genes > 0 else 0.0,
                'active_cells': sum(1 for cell in island_data['cells'] if cell['fitness'] > 0),
                'cell_activation_rate': sum(1 for cell in island_data['cells'] if cell['fitness'] > 0) / len(island_data['cells']),
                'diversity_score': _calculate_population_diversity(island_data['cells'])
            })
        
        # Cell type distribution KPIs
        cell_types = {
            'stem': sum(1 for cell in island_data['cells'] if cell['type'] == 'stem'),
            'biosensor': sum(1 for cell in island_data['cells'] if cell['type'] == 'biosensor'),
            'effector': sum(1 for cell in island_data['cells'] if cell['type'] == 'effector'),
            'controller': sum(1 for cell in island_data['cells'] if cell['type'] == 'controller'),
            'quantum': sum(1 for cell in island_data['cells'] if cell['type'] == 'quantum'),
            'balanced': sum(1 for cell in island_data['cells'] if cell['type'] == 'balanced')
        }
        island_kpis['cell_type_distribution'] = cell_types
        
        # Gene type distribution KPIs
        gene_types = defaultdict(int)
        for cell in island_data['cells']:
            for gene in cell.get('genes', []):
                if gene.get('is_active', False):
                    gene_types[gene.get('gene_type', 'unknown')] += 1
        
        island_kpis['gene_type_distribution'] = dict(gene_types)
        
        # Hall of Fame KPIs
        if island_data['hall_of_fame']:
            hof_fitnesses = [entry['fitness'] for entry in island_data['hall_of_fame']]
            island_kpis.update({
                'hall_of_fame_size': len(island_data['hall_of_fame']),
                'hof_mean_fitness': sum(hof_fitnesses) / len(hof_fitnesses),
                'hof_max_fitness': max(hof_fitnesses),
                'hof_fitness_std': np.std(hof_fitnesses) if len(hof_fitnesses) > 1 else 0.0,
                'hof_fitness_range': max(hof_fitnesses) - min(hof_fitnesses)
            })
        
        # Stress and system state KPIs
        island_kpis.update({
            'current_stress': getattr(island, 'current_stress', 0.0),
            'generation': getattr(island, 'generation', generation),
            'island_type': getattr(island, 'island_type', 'unknown'),
            'device': str(getattr(island, 'device', 'unknown'))
        })
        
        # PNN-specific KPIs if available
        if hasattr(island, 'pnn_state'):
            pnn_state = island.pnn_state
            island_kpis.update({
                'pnn_locked_cells': getattr(pnn_state, 'locked_cells', 0),
                'pnn_unlocked_cells': getattr(pnn_state, 'unlocked_cells', 0),
                'pnn_lock_rate': getattr(pnn_state, 'lock_rate', 0.0),
                'pnn_stability_score': getattr(pnn_state, 'stability_score', 0.0)
            })
        
        # Circadian rhythm KPIs if available
        if hasattr(island, 'circadian_state'):
            circadian_state = island.circadian_state
            island_kpis.update({
                'circadian_phase': getattr(circadian_state, 'current_phase', 0.0),
                'circadian_plasticity_multiplier': getattr(circadian_state, 'plasticity_multiplier', 1.0),
                'circadian_gate_open': getattr(circadian_state, 'gate_open', True)
            })
        
        # Reaction-diffusion KPIs if available
        if hasattr(island, 'rd_stress_field'):
            rd_field = island.rd_stress_field
            island_kpis.update({
                'rd_mean_stress': getattr(rd_field, 'mean_stress', 0.0),
                'rd_max_stress': getattr(rd_field, 'max_stress', 0.0),
                'rd_stress_std': getattr(rd_field, 'stress_std', 0.0),
                'rd_unlock_events': getattr(rd_field, 'unlock_events', 0),
                'rd_transposition_events': getattr(rd_field, 'transposition_events', 0)
            })
        
        # Architecture evolution KPIs
        total_modifications = sum(
            cell.get('architecture', {}).get('modifications', 0) 
            for cell in island_data['cells'] 
            if cell.get('architecture')
        )
        island_kpis['total_architecture_modifications'] = total_modifications
        
        # Add any existing KPIs from the island
        if hasattr(island, 'generation_kpis') and island.generation_kpis:
            island_kpis.update(island.generation_kpis)
        
        island_data['kpis'] = island_kpis
        
        islands_data.append(island_data)
    
    # Collect Causal Tapestry data
    causal_tapestry_data = {}
    if hasattr(archipelago, 'causal_tapestry') and archipelago.causal_tapestry:
        tapestry = archipelago.causal_tapestry
        if hasattr(tapestry, 'graph') and tapestry.graph:
            causal_tapestry_data = {
                'total_nodes': tapestry.graph.number_of_nodes(),
                'total_edges': tapestry.graph.number_of_edges(),
                'run_id': getattr(tapestry, 'run_id', run_id),  # Use the current run_id
                'recent_events': [],
                'lineage_depth': 0,
                'event_types': {}
            }

            # Get recent events and analyze event types
            event_type_counts = {}
            for node, data in tapestry.graph.nodes(data=True):
                if data.get('type') == 'event':
                    event_type = data.get('event_type', 'unknown')
                    event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

                    if data.get('generation', 0) >= generation - 10:
                        event_info = {
                            'event_id': str(node),
                            'event_type': event_type,
                            'generation': data.get('generation', 0),
                            'effect': data.get('effect', 0.0),
                            'details': data.get('details', {})
                        }
                        causal_tapestry_data['recent_events'].append(event_info)

            causal_tapestry_data['event_types'] = event_type_counts
            
            # Calculate lineage depth
            max_depth = 0
            for node, data in tapestry.graph.nodes(data=True):
                if data.get('type') == 'cell':
                    try:
                        ancestors = list(nx.ancestors(tapestry.graph, node))
                        if ancestors:
                            depth = len(ancestors)
                            max_depth = max(max_depth, depth)
                    except:
                        pass
            causal_tapestry_data['lineage_depth'] = max_depth

            # Sort events by generation (most recent first)
            causal_tapestry_data['recent_events'].sort(
                key=lambda x: x['generation'], reverse=True
            )
            causal_tapestry_data['recent_events'] = causal_tapestry_data['recent_events'][:20]
    
    # Get complete configuration
    complete_config = get_config_dict(cfg)
    
    # Legacy factors dict for backward compatibility
    factors = {
        'pnn': getattr(cfg, 'enable_pnn', False),
        'rd': getattr(cfg, 'enable_reaction_diffusion_stress', False),
        'circadian': getattr(cfg, 'enable_circadian_rhythm', False),
        'immune': getattr(cfg, 'enable_stress_immune', False),
        'quantization': getattr(cfg, 'enable_quantization', False),
        'static_resource_partitioning': getattr(cfg, 'enable_static_resource_partitioning', False),
        'cuda_allocator_conf': getattr(cfg, 'enable_cuda_allocator_conf', False),
        'telemetry': getattr(cfg, 'enable_telemetry', True),
        'microchimerism': getattr(cfg, 'microchimerism_seeding_rate', 0) > 0
    }
    hardware = {
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda': torch.version.cuda
    }

    # Fetch KPIs dictionary
    kpis = getattr(archipelago, 'generation_kpis', {})

    # Create the complete state object
    state = {
        'run_id': run_id,
        'factors': factors,  # Legacy factors for backward compatibility
        'configuration': complete_config,  # Complete configuration settings
        'hardware': hardware,
        'kpis': kpis,
        'archipelago_summary': archipelago_summary,
        'islands': islands_data,
        'causal_tapestry': causal_tapestry_data,
        'generation': generation,
        'timestamp': time.time()
    }
    
    # Set up file paths using the run_id
    viz_dir = os.path.join(cfg.telemetry_dir, run_id)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Write the consolidated file
    unique_filename = os.path.join(viz_dir, f"generation_{generation:04d}_archipelago_state.json")
    
    logger.info(f"📁 Writing archipelago state to: {unique_filename}")
    
    # Initialize async telemetry if needed
    if _telemetry_queue is None:
        init_async_telemetry()
    
    # Use async writing if available
    if _telemetry_queue:
        try:
            # Queue the write operation
            _telemetry_queue.put_nowait((unique_filename, state))
            logger.info(f"✅ Queued archipelago state file for async writing")
        except queue.Full:
            # Fallback to sync writing if queue is full
            logger.warning("[TELEMETRY] Queue full, falling back to sync write")
            try:
                with state_lock:
                    atomic_json_write(state, unique_filename, ensure_ascii=False, indent=2)
                logger.info(f"✅ Successfully wrote archipelago state file (sync)")
            except Exception as e:
                logger.error(f"❌ Failed to write archipelago state file: {e}")
    else:
        # Sync writing fallback
        try:
            with state_lock:
                atomic_json_write(state, unique_filename, ensure_ascii=False, indent=2)
            logger.info(f"✅ Successfully wrote archipelago state file")
        except Exception as e:
            logger.error(f"❌ Failed to write archipelago state file: {e}")
        raise
    
    logger.info(f"📊 Archipelago visualization state written for generation {generation} with {len(islands_data)} islands")
    
    return unique_filename




class TermColors:
    """Utility class for terminal colors and styles."""
    # Basic Colors
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright Colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\032[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Styles
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'






def _event_emitter():
    """Get the event emitter"""
    global _event_emitter
    return _event_emitter

def set_germinal_center(gc):
    """Set the global germinal center reference"""
    global _current_germinal_center
    _current_germinal_center = gc


def _calculate_population_diversity(cells_data):
    """Calculate diversity metric for the population"""
    if not cells_data:
        return 0.0
    
    # Gene type diversity
    gene_types = defaultdict(int)
    total_genes = 0
    for cell in cells_data:
        for gene in cell.get('genes', []):
            if gene.get('is_active'):
                gene_types[gene.get('gene_type', 'unknown')] += 1
                total_genes += 1
    
    if total_genes == 0:
        return 0.0
    
    # Shannon entropy for gene type diversity
    entropy = 0
    for count in gene_types.values():
        if count > 0:
            p = count / total_genes
            entropy -= p * np.log(p + 1e-10)
    
    # Normalize by max possible entropy
    max_entropy = np.log(len(gene_types)) if len(gene_types) > 0 else 1
    diversity = entropy / max_entropy if max_entropy > 0 else 0
    
    return diversity





def _write_single_cell_architecture_state(cell_id, architecture_modifier, island_name=None):
    """Writes detailed architecture state for a single cell to a unique file."""
    print("Writing enhanced visualization state")
    # Capture current architecture state
    architecture_state = {
        'modules': {},
        'connections': dict(architecture_modifier.module_connections),
        'modification_history': []
    }
    
    # Analyze each module in detail
    for name, module in architecture_modifier.dynamic_modules.items():
        module_info = {
            'name': name,
            'type': 'sequential' if isinstance(module, nn.Sequential) else 'linear',
            'layers': [],
            'position': _calculate_module_position(name, architecture_modifier),
            'size': 0,
            'activation': None,
            'color': '#4A90E2'  # Default blue
        }
        
        if isinstance(module, nn.Sequential):
            # Analyze Sequential module structure
            for i, layer in enumerate(module):
                layer_info = {
                    'index': i,
                    'type': type(layer).__name__,
                    'params': {}
                }
                
                if isinstance(layer, nn.Linear):
                    layer_info['params'] = {
                        'in_features': layer.in_features,
                        'out_features': layer.out_features
                    }
                    module_info['size'] = layer.out_features
                elif isinstance(layer, nn.LayerNorm):
                    layer_info['params'] = {
                        'normalized_shape': layer.normalized_shape
                    }
                elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.ELU, nn.GELU, nn.SiLU)):
                    module_info['activation'] = type(layer).__name__.lower().replace('silu', 'swish')
                    module_info['color'] = _get_activation_color(type(layer).__name__)
                elif isinstance(layer, nn.Dropout):
                    layer_info['params'] = {
                        'p': layer.p
                    }
                
                module_info['layers'].append(layer_info)
        
        elif isinstance(module, nn.Linear):
            # Handle standalone Linear module (like output)
            module_info['size'] = module.out_features
            module_info['layers'] = [{
                'index': 0,
                'type': 'Linear',
                'params': {
                    'in_features': module.in_features,
                    'out_features': module.out_features
                }
            }]
        
        architecture_state['modules'][name] = module_info
    
    # Process modification history and emit events
    
    # Get population data - capture ALL cells like in _write_population_visualization_state
    cells_data = []
    generation = 0
    hall_of_fame_data = []
    causal_tapestry_data = {}
    
    if _current_germinal_center and hasattr(_current_germinal_center, 'population'):
        generation = getattr(_current_germinal_center, 'generation', 0)
        
        # Capture Hall of Fame data
        if hasattr(_current_germinal_center, 'hall_of_fame') and _current_germinal_center.hall_of_fame:
            for idx, (fitness, champion_cell) in enumerate(_current_germinal_center.hall_of_fame):
                hall_of_fame_entry = {
                    'rank': idx + 1,
                    'fitness': float(fitness),
                    'cell_id': str(champion_cell.cell_id),
                    'generation': getattr(champion_cell, 'generation', 0),
                    'lineage': getattr(champion_cell, 'lineage', []),
                    'island': getattr(_current_germinal_center, 'island_desc', 'Unknown')
                }
                # Add gene information for hall of fame champions
                if hasattr(champion_cell, 'genes'):
                    hall_of_fame_entry['genes'] = [
                        {
                            'gene_id': str(getattr(gene, 'gene_id', str(id(gene)))),
                            'gene_type': str(getattr(gene, 'gene_type', 'V')),
                            'is_active': bool(getattr(gene, 'is_active', False)),
                            'is_quantum': 'Quantum' in gene.__class__.__name__
                        }
                        for gene in champion_cell.genes
                    ]
                hall_of_fame_data.append(hall_of_fame_entry)
        
        # Capture Causal Tapestry data if available
        if hasattr(_current_germinal_center, 'causal_tapestry'):
            tapestry = _current_germinal_center.causal_tapestry
            if hasattr(tapestry, 'graph') and tapestry.graph:
                causal_tapestry_data = {
                    'total_nodes': tapestry.graph.number_of_nodes(),
                    'total_edges': tapestry.graph.number_of_edges(),
                    'run_id': getattr(tapestry, 'run_id', run_id),
                    'recent_events': []
                }
                
                # Get recent events (last 10 generations)
                current_gen = generation
                for node, data in tapestry.graph.nodes(data=True):
                    if (data.get('type') == 'event' and 
                        data.get('generation', 0) >= current_gen - 10):
                        event_info = {
                            'event_id': str(node),
                            'event_type': data.get('event_type', 'unknown'),
                            'generation': data.get('generation', 0),
                            'effect': data.get('effect', 0.0),
                            'details': data.get('details', {})
                        }
                        causal_tapestry_data['recent_events'].append(event_info)
                
                # Sort events by generation (most recent first)
                causal_tapestry_data['recent_events'].sort(
                    key=lambda x: x['generation'], reverse=True
                )
                causal_tapestry_data['recent_events'] = causal_tapestry_data['recent_events'][:20]  # Limit to 20 most recent
        
        # Capture ALL cells in the population
        for idx, (cid, cell) in enumerate(list(_current_germinal_center.population.items())):
            # Determine cell type
            type_counts = defaultdict(int)
            cell_type = 'balanced'
            
            if hasattr(cell, 'genes'):
                active_genes = [g for g in cell.genes if g.is_active]
                if active_genes:
                    for gene in active_genes:
                        type_counts[gene.gene_type] += 1
                    
                    if type_counts:
                        dominant_type = max(type_counts, key=type_counts.get)
                        type_mapping = {
                            'S': 'stem',
                            'V': 'biosensor',
                            'D': 'effector',
                            'J': 'controller',
                            'Q': 'quantum'
                        }
                        cell_type = type_mapping.get(dominant_type, 'balanced')
            
            cell_info = {
                'cell_id': cid,
                'index': idx,
                'fitness': cell.fitness_history[-1] if cell.fitness_history else 0.0,
                'generation': getattr(cell, 'generation', generation),
                'lineage': getattr(cell, 'lineage', []),
                'type': cell_type, # <-- The calculated type is added here
                'genes': [],
                'architecture': None,
                'connections': []
            }
            # Collect gene information
            if hasattr(cell, 'genes'):
                for gene in cell.genes:
                    gene_info = {
                        'gene_id': str(getattr(gene, 'gene_id', str(id(gene)))),
                        'gene_type': str(getattr(gene, 'gene_type', 'V')),
                        'position': int(getattr(gene, 'position', 0)),
                        'is_active': bool(getattr(gene, 'is_active', False)),
                        'is_quantum': 'Quantum' in gene.__class__.__name__,
                        'depth': float(gene.compute_depth().item()) if hasattr(gene, 'compute_depth') else 1.0,
                        'activation': float(getattr(gene, 'activation_ema', 0.0)),
                        'variant_id': int(getattr(gene, 'variant_id', 0)),
                        'methylation': float(gene.methylation_state.mean().item()) if hasattr(gene, 'methylation_state') else 0.0
                    }
                    cell_info['genes'].append(gene_info)
                
                # Track gene connections
                active_genes = [g for g in cell.genes if g.is_active]
                for idx1, gene1 in enumerate(active_genes):
                    for idx2, gene2 in enumerate(active_genes[idx1+1:], idx1+1):
                        cell_info['connections'].append({
                            'source': str(gene1.gene_id),
                            'target': str(gene2.gene_id),
                            'strength': float(abs(idx1 - idx2) / len(active_genes)) if active_genes else 0.0
                        })
            
            # Add architecture information if this is the current cell
            if cid == cell_id:
                cell_info['architecture'] = architecture_state
            
            cells_data.append(cell_info)
    
    # Get complete configuration
    complete_config = get_config_dict(cfg)
    
    # Create visualization state
    state = {
        'timestamp': time.time(),
        'generation': generation,
        'current_cell_id': cell_id,
        'cells': cells_data,
        'architecture_state': architecture_state,
        'hall_of_fame': hall_of_fame_data,
        'causal_tapestry': causal_tapestry_data,
        'configuration': complete_config  # Add complete configuration
    }
    
    # Ensure visualization directory exists
    run_id = get_or_create_run_id()
    
    # Create island-specific directory if island_name is provided
    if island_name:
        # Sanitize island name for directory creation
        safe_island_name = "".join(c for c in island_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_island_name = safe_island_name.replace(' ', '_')
        viz_dir = os.path.join(cfg.telemetry_dir, run_id, safe_island_name)
    else:
        viz_dir = os.path.join(cfg.telemetry_dir, run_id)
    
    os.makedirs(viz_dir, exist_ok=True)

    # Write state to unique file
    unique_filename = os.path.join(viz_dir, f"architecture_cell_{cell_id}_gen_{generation}.json")
    
    with state_lock:
        # Write to unique filename for archival
        atomic_json_write(state, unique_filename, ensure_ascii=False, indent=2)
        logger.debug(f"Wrote single cell architecture state to {unique_filename}")
        
        # Write island-specific architecture state if island_name provided
        if island_name:
            island_arch_path = os.path.join(viz_dir, f"{safe_island_name}_architecture_state.json")
            atomic_json_write(state, island_arch_path, ensure_ascii=False, indent=2)
            logger.debug(f"Updated island architecture state at {island_arch_path}")
        
        # Write to global architecture state files
        architecture_json_path = os.path.join(cfg.telemetry_dir, "te_ai_state.json")
        atomic_json_write(state, architecture_json_path, ensure_ascii=False, indent=2)
        logger.debug(f"Updated architecture_state.json at {architecture_json_path}")
        architecture_json_path = os.path.join(cfg.telemetry_dir, 'architecture_state.json')
        atomic_json_write(state, architecture_json_path, ensure_ascii=False, indent=2)
        logger.debug(f"Updated architecture_state.json at {architecture_json_path}")


def _calculate_module_position(module_name, architecture_modifier):
    """Calculate 3D position based on connection graph"""
    # Build graph for topological analysis
    connections = architecture_modifier.module_connections
    all_modules = set(architecture_modifier.dynamic_modules.keys())
    
    # Find layers with no incoming connections (input layers)
    input_layers = all_modules - set(sum(connections.values(), []))
    
    # Calculate layer depth (distance from input)
    depths = {}
    visited = set()
    queue = [(layer, 0) for layer in input_layers]
    
    while queue:
        current, depth = queue.pop(0)
        if current in visited:
            continue
        
        visited.add(current)
        depths[current] = depth
        
        # Add connected layers
        if current in connections:
            for next_layer in connections[current]:
                if next_layer not in visited:
                    queue.append((next_layer, depth + 1))
    
    # Get depth for this module
    depth = depths.get(module_name, 0)
    
    # Calculate vertical position based on parallel paths
    layers_at_depth = [m for m, d in depths.items() if d == depth]
    y_offset = layers_at_depth.index(module_name) if module_name in layers_at_depth else 0
    
    # Calculate Z position based on connectivity
    incoming = sum(1 for k, v in connections.items() if module_name in v)
    outgoing = len(connections.get(module_name, []))
    
    return {
        'x': depth * 3.0,
        'y': y_offset * 2.0 - len(layers_at_depth) / 2.0,
        'z': (outgoing - incoming) * 0.5
    }

def _get_activation_color(activation_name):
    """Get color based on activation type"""
    colors = {
        'ReLU': '#FF6B6B',
        'Tanh': '#4ECDC4',
        'Sigmoid': '#F7DC6F',
        'ELU': '#BB8FCE',
        'GELU': '#85C1E2',
        'SiLU': '#50E3C2',  # Swish
        'Linear': '#FFFFFF'
    }
    return colors.get(activation_name, '#9013FE')




def _write_population_visualization_state(cell_id, architecture_modifier, island_name=None):
    """Writes a comprehensive visualization state including the entire population,
    Hall of Fame, Causal Tapestry summary, and detailed architecture for the champion cell."""

    cells_data = []
    population = None
    generation = 0
    current_stress = 0
    hall_of_fame_data = []
    causal_tapestry_data = {}

    # Try to access the germinal center instance
    global _current_germinal_center
    if _current_germinal_center and hasattr(_current_germinal_center, 'population'):
        population = _current_germinal_center.population
        generation = getattr(_current_germinal_center, 'generation', 0)
        current_stress = getattr(_current_germinal_center, 'current_stress', 0)
        
        # Capture Hall of Fame data
        if hasattr(_current_germinal_center, 'hall_of_fame') and _current_germinal_center.hall_of_fame:
            for idx, (fitness, champion_cell) in enumerate(_current_germinal_center.hall_of_fame):
                hall_of_fame_entry = {
                    'rank': idx + 1,
                    'fitness': float(fitness),
                    'cell_id': str(champion_cell.cell_id),
                    'generation': getattr(champion_cell, 'generation', 0),
                    'lineage': getattr(champion_cell, 'lineage', []),
                    'island': getattr(_current_germinal_center, 'island_desc', 'Unknown')
                }
                if hasattr(champion_cell, 'genes'):
                    hall_of_fame_entry['genes'] = [
                        {
                            'gene_id': str(getattr(gene, 'gene_id', str(id(gene)))),
                            'gene_type': str(getattr(gene, 'gene_type', 'V')),
                            'is_active': bool(getattr(gene, 'is_active', False)),
                            'is_quantum': 'Quantum' in gene.__class__.__name__
                        }
                        for gene in champion_cell.genes
                    ]
                hall_of_fame_data.append(hall_of_fame_entry)
        
        # Capture Causal Tapestry data
        tapestry = None
        if hasattr(_current_germinal_center, 'causal_tapestry'):
            tapestry = _current_germinal_center.causal_tapestry
        elif hasattr(_current_germinal_center, 'archipelago') and hasattr(_current_germinal_center.archipelago, 'causal_tapestry'):
            tapestry = _current_germinal_center.archipelago.causal_tapestry

        if tapestry and hasattr(tapestry, 'graph') and tapestry.graph:
            causal_tapestry_data = {
                'total_nodes': tapestry.graph.number_of_nodes(),
                'total_edges': tapestry.graph.number_of_edges(),
                'run_id': getattr(tapestry, 'run_id', run_id),
                'recent_events': [],
                'lineage_depth': 0,
                'event_types': {}
            }

            # Get recent events and analyze event types
            current_gen = generation
            event_type_counts = {}

            for node, data in tapestry.graph.nodes(data=True):
                if data.get('type') == 'event':
                    event_type = data.get('event_type', 'unknown')
                    event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

                    if data.get('generation', 0) >= current_gen - 10:
                        event_info = {
                            'event_id': str(node),
                            'event_type': event_type,
                            'generation': data.get('generation', 0),
                            'effect': data.get('effect', 0.0),
                            'details': data.get('details', {})
                        }
                        causal_tapestry_data['recent_events'].append(event_info)

            causal_tapestry_data['event_types'] = event_type_counts
            max_depth = 0
            for node, data in tapestry.graph.nodes(data=True):
                if data.get('type') == 'cell':
                    try:
                        # Find the longest path from any root to this node
                        ancestors = list(nx.ancestors(tapestry.graph, node))
                        if ancestors:
                            depth = len(ancestors)
                            max_depth = max(max_depth, depth)
                    except:
                        pass  # Skip if graph analysis fails
            causal_tapestry_data['lineage_depth'] = max_depth

            # Sort events by generation (most recent first)
            causal_tapestry_data['recent_events'].sort(
                key=lambda x: x['generation'], reverse=True
            )
            causal_tapestry_data['recent_events'] = causal_tapestry_data['recent_events'][:20]  # Limit to 20 most recent

    if population:
        for idx, (cid, cell) in enumerate(list(population.items())):
            # Determine cell type based on dominant active genes
            type_counts = defaultdict(int)
            cell_type = 'balanced'

            if hasattr(cell, 'genes'):
                active_genes = [g for g in cell.genes if g.is_active]
                if active_genes:
                    for gene in active_genes:
                        type_counts[gene.gene_type] += 1
                    
                    if type_counts:
                        dominant_type = max(type_counts, key=type_counts.get)
                        
                        type_mapping = {
                            'S': 'stem',
                            'V': 'biosensor',
                            'D': 'effector',
                            'J': 'controller',
                            'Q': 'quantum'
                        }
                        cell_type = type_mapping.get(dominant_type, 'balanced')

            cell_info = {
                'cell_id': cid,
                'index': idx,
                'fitness': cell.fitness_history[-1] if cell.fitness_history else 0.0,
                'generation': getattr(cell, 'generation', generation),
                'lineage': getattr(cell, 'lineage', []),
                'type': cell_type, # <-- The calculated type is added here
                'genes': [],
                'architecture': None,
                'connections': []
            }
            
            # Collect gene information
            if hasattr(cell, 'genes'):
                for gene in cell.genes:
                    gene_info = {
                        'gene_id': str(getattr(gene, 'gene_id', str(id(gene)))),
                        'gene_type': str(getattr(gene, 'gene_type', 'V')),
                        'position': int(getattr(gene, 'position', 0)),
                        'is_active': bool(getattr(gene, 'is_active', False)),
                        'is_quantum': 'Quantum' in gene.__class__.__name__,
                        'depth': float(gene.compute_depth().item()) if hasattr(gene, 'compute_depth') else 1.0,
                        'activation': float(getattr(gene, 'activation_ema', 0.0)),
                        'variant_id': int(getattr(gene, 'variant_id', 0)),
                        'methylation': float(gene.methylation_state.mean().item()) if hasattr(gene, 'methylation_state') else 0.0
                    }
                    cell_info['genes'].append(gene_info)
                
                # Track gene connections
                active_genes = [g for g in cell.genes if g.is_active]
                for idx1, gene1 in enumerate(active_genes):
                    for idx2, gene2 in enumerate(active_genes[idx1+1:], idx1+1):
                        cell_info['connections'].append({
                            'source': str(gene1.gene_id),
                            'target': str(gene2.gene_id),
                            'strength': float(abs(idx1 - idx2) / len(active_genes)) if active_genes else 0.0
                        })
            
            # Add architecture information if this is the current cell
            if cid == cell_id and hasattr(cell, 'architecture_modifier'):
                arch = cell.architecture_modifier
                try:
                    # Safely extract serializable architecture info
                    module_names = []
                    if hasattr(arch, 'dynamic_modules'):
                        module_names = list(arch.dynamic_modules.keys())
                    
                    connections = {}
                    if hasattr(arch, 'module_connections'):
                        # Convert defaultdict to regular dict and ensure all values are lists
                        for k, v in list(arch.module_connections.items()):
                            connections[str(k)] = list(v) if isinstance(v, (list, set, tuple)) else [str(v)]
                    
                    cell_info['architecture'] = {
                        'dna': str(getattr(arch, 'architecture_dna', 'N/A')),
                        'modules': module_names,
                        'connections': connections,
                        'modifications': len(getattr(arch, 'modification_history', []))
                    }
                except Exception as arch_error:
                    logger.warning(f"Failed to serialize architecture info: {arch_error}")
                    cell_info['architecture'] = {
                        'dna': 'error',
                        'modules': [],
                        'connections': {},
                        'modifications': 0
                    }
            
            cells_data.append(cell_info)

    # Get complete configuration
    complete_config = get_config_dict(cfg)
    
    state = {
        'cells': cells_data,
        'cell_id': cell_id,
        'generation': generation,
        'population_size': len(population) if population else 1,
        'total_genes': sum(len(c.get('genes', [])) for c in cells_data),
        'active_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('is_active', False)),
        'quantum_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('is_quantum', False)),
        'cell_types': {
            'V_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('gene_type') == 'V' and g.get('is_active')),
            'D_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('gene_type') == 'D' and g.get('is_active')),
            'J_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('gene_type') == 'J' and g.get('is_active')),
            'Q_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('is_quantum', False) and g.get('is_active')),
            'S_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('gene_type') == 'S' and g.get('is_active'))
        },
        'phase': 'normal',
        'stress_level': current_stress,
        'mean_fitness': sum(c.get('fitness', 0) for c in cells_data) / max(len(cells_data), 1),
        'hall_of_fame_size': len(hall_of_fame_data),
        'best_fitness': hall_of_fame_data[0]['fitness'] if hall_of_fame_data else 0.0,
        'causal_tapestry': causal_tapestry_data,
        'configuration': complete_config,  # Add complete configuration
        'timestamp': time.time()
    }

    run_id = get_or_create_run_id()
        
    if island_name:
        safe_island_name = "".join(c for c in island_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_island_name = safe_island_name.replace(' ', '_')
        viz_dir = os.path.join(cfg.telemetry_dir, run_id, safe_island_name)
    else:
        viz_dir = os.path.join(cfg.telemetry_dir, run_id)
    
    os.makedirs(viz_dir, exist_ok=True)

    if island_name:
        unique_filename = os.path.join(viz_dir, f"generation_{generation:04d}_state.json")
        realtime_filename = os.path.join(viz_dir, f"{safe_island_name}_current_state.json")
    else:
        unique_filename = os.path.join(viz_dir, f"generation_{generation:04d}_state.json")

    with state_lock:
        atomic_json_write(state, unique_filename, ensure_ascii=False, indent=2)

        if island_name:
            atomic_json_write(state, realtime_filename, ensure_ascii=False, indent=2)
        
        atomic_json_write(state, os.path.join(cfg.telemetry_dir, 'te_ai_state.json'), ensure_ascii=False)
            
        pointer = {
            'current_run_id': run_id,
            'current_generation': generation,
            'data_directory': viz_dir,
            'latest_state_file': unique_filename,
            'te_ai_state_file': os.path.join(cfg.telemetry_dir, 'te_ai_state.json'),  # For live polling
            'timestamp': time.time()
        }
        atomic_json_write(pointer, os.path.join(cfg.telemetry_dir, 'current_run_pointer.json'), ensure_ascii=False, indent=2)
