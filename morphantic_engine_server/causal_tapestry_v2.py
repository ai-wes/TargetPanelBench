import numpy as np
from threading import RLock, Thread
from queue import Queue, Empty
from datetime import datetime
import json
from pathlib import Path
import random
from typing import Dict, Any, List, Tuple, Optional

# (Optional but recommended for async logging)
# from queue import Queue, Empty
# from threading import Thread

class CausalTapestry:
    """
    A high-performance, shared knowledge store that tracks the lineage, genetics,
    and key events of the entire Symbiotic Swarm.
    
    RE-IMPLEMENTATION (2025-08-12): Replaced the networkx backend with native Python
    dictionaries for a significant performance increase in high-frequency logging
    and querying operations, eliminating major computational overhead. The public
    API remains identical to the original implementation.
    """
    def __init__(self):
        # --- NEW: Dictionary-based graph representation ---
        self.nodes: Dict[str, Dict[str, Any]] = {}  # {node_id: data_dict}
        self.edges: Dict[str, Dict[str, List[str]]] = {} # {node_id: {'parents': [], 'children': []}}
        self.edge_attrs: Dict[tuple, Dict[str, Any]] = {}  # {(u,v): {type, role, ...}}
        
        self.run_id: Optional[str] = None
        self.event_timestamps: Dict[str, float] = {}
        self.max_graph_size: int = 10000
        self.prune_target_fraction: float = 0.7
        self.generation_prune_enabled: bool = True
        self.max_generation_age: Optional[int] = 200
        self.prune_check_interval_gens: int = 10
        self._last_prune_gen: int = 0
        self._max_seen_generation: int = 0

        # --- Directional memory and other features remain the same ---
        self._ctx_maps = {'action': {}, 'island': {}, 'pnn_state': {}, 'parent_types': {}}
        self._ctx_next_id = {'action': 0, 'island': 0, 'pnn_state': 0, 'parent_types': 0}
        self.direction_stats: dict[tuple[int, int], dict] = {}
        self.PROJ_DIM: int = 32
        self.RING_SIZE: int = 32
        self._proj_seed: int = 42
        self._proj_cache: dict[int, np.ndarray] = {}
        self.effect_good_threshold: float = -1e-3
        self.effect_bad_threshold: float = 1e-3
        self.log_only_extremes: bool = True
        self.directional_effects: Dict[str, Dict[tuple, List[np.ndarray]]] = {}
        self._direction_cache: Dict[tuple, np.ndarray] = {}
        self.direction_max_samples_per_context: Optional[int] = None
        self.enable_events: bool = True
        self.event_sampling_prob: float = 1.0
        self.compact_event_details: bool = True
        self._lock: RLock = RLock()

        # --- Embeddings / vector store (NodeVec-v1 / EdgeVec-v1) ---
        self.EMBED_VERSION: str = "NodeVec-v1"
        self.HASH_DIM: int = 128
        self.node_vecs_fp16: Dict[str, np.ndarray] = {}
        self.edge_vecs_fp16: Dict[tuple, np.ndarray] = {}
        self._embed_dirty_nodes: set[str] = set()
        self._embed_dirty_edges: set[tuple] = set()

        # --- (Optional) Asynchronous Logging Setup ---
        # self._log_queue = Queue()
        # self._stop_event = threading.Event()
        # self._log_thread = Thread(target=self._process_log_queue, daemon=True)
        # self._log_thread.start()

    # def _process_log_queue(self):
    #     """Worker thread to process logging events asynchronously."""
    #     while not self._stop_event.is_set():
    #         try:
    #             # Wait for up to 1 second for an item
    #             method_name, args, kwargs = self._log_queue.get(timeout=1)
    #             method = getattr(self, f"_sync_{method_name}")
    #             method(*args, **kwargs)
    #         except Empty:
    #             continue
    #         except Exception as e:
    #             print(f"Error in CausalTapestry log thread: {e}")

    def reset(self, run_id: str):
        with self._lock:
            self.nodes.clear()
            self.edges.clear()
            self.event_timestamps.clear()
            self.run_id = run_id
        print(f"Causal Tapestry reset for new run: {run_id}")

    def _ensure_node_edges(self, node_id: str):
        """Helper to initialize edge structure for a node if it doesn't exist."""
        if node_id not in self.edges:
            self.edges[node_id] = {'parents': [], 'children': []}

    def add_cell_node(self, cell_id: str, generation: int, island_name: str, fitness: float, genes: list):
        with self._lock:
            self._max_seen_generation = max(self._max_seen_generation, generation)
            node_data = {
                'type': 'cell',
                'generation': generation,
                'island': island_name,
                'fitness': fitness,
                'genes': ",".join(map(str, genes))
            }
            self.nodes[cell_id] = node_data
            self._ensure_node_edges(cell_id)
            self._mark_node_dirty(cell_id)
        
        self._prune_graph_if_needed()
        self._prune_by_generation_if_needed(generation)

    def add_gene_node(self, gene_id: str, gene_type: str, variant_id: Any):
        with self._lock:
            if gene_id not in self.nodes:
                self.nodes[gene_id] = {
                    'type': 'gene',
                    'gene_type': gene_type,
                    'variant_id': str(variant_id)
                }
                self._ensure_node_edges(gene_id)
                self._mark_node_dirty(gene_id)
        self._prune_graph_if_needed()

    def add_event_node(self, event_id: str, event_type: str, generation: int, details: Dict):
        if not self.enable_events or (self.event_sampling_prob < 1.0 and random.random() > self.event_sampling_prob):
            return

        eff_val = float(details.get('effect', 0.0))
        if self.log_only_extremes and not (eff_val <= self.effect_good_threshold or eff_val >= self.effect_bad_threshold):
            return

        timestamp = datetime.now().timestamp()
        with self._lock:
            cdetails = details
            if self.compact_event_details:
                allowed = {'action', 'effect', 'island', 'pnn_state', 'stress_bin', 'parent_types', 'child_has_quantum', 'strategy_used'}
                cdetails = {k: v for k, v in details.items() if k in allowed}

            self._max_seen_generation = max(self._max_seen_generation, generation)
            self.nodes[event_id] = {
                'type': 'event',
                'event_type': event_type,
                'generation': generation,
                'details': json.dumps(cdetails),
                'effect': eff_val,
                'timestamp': timestamp
            }
            self._ensure_node_edges(event_id)
            self.event_timestamps[event_id] = timestamp
            self._mark_node_dirty(event_id)
        
        self._prune_graph_if_needed()
        self._prune_by_generation_if_needed(generation)

        # Directional stats and memory logic remains the same, as it's not tied to networkx
        if event_type == 'MUTATION' and details.get('action') in ('recombine', 'mutate'):
            vec = details.get('mutation_vector')
            if vec is not None:
                self._update_direction_stats(self._get_action_id(details['action']), self._get_context_id(details), np.asarray(vec, dtype=np.float32), eff_val)
                if eff_val < 0.0:
                    self._store_directional_effect(details['action'], details, vec)

    def _store_directional_effect(self, action, details, vec):
        """Helper for storing successful mutation vectors."""
        context_key = self._make_context_key(details)
        if action not in self.directional_effects:
            self.directional_effects[action] = {}
        bucket = self.directional_effects[action].setdefault(context_key, [])
        bucket.append(np.asarray(vec, dtype=float))
        if self.direction_max_samples_per_context is not None and len(bucket) > self.direction_max_samples_per_context:
            del bucket[:len(bucket) - self.direction_max_samples_per_context]
        self._direction_cache.pop((action, context_key), None)

    def log_lineage(self, parent_id: str, child_id: str):
        with self._lock:
            self._ensure_node_edges(parent_id)
            self._ensure_node_edges(child_id)
            if child_id not in self.edges[parent_id]['children']:
                self.edges[parent_id]['children'].append(child_id)
            if parent_id not in self.edges[child_id]['parents']:
                self.edges[child_id]['parents'].append(parent_id)
            self.edge_attrs[(parent_id, child_id)] = {'type': 'lineage', 'role': None}
            self._mark_edge_dirty(parent_id, child_id)
            self._mark_node_dirty(parent_id); self._mark_node_dirty(child_id)

    # Typed edge logging with attributes
    def log_gene_composition(self, cell_id: str, gene_id: str):
        with self._lock:
            self._ensure_node_edges(cell_id)
            self._ensure_node_edges(gene_id)
            # model composition as edge gene -> cell
            if cell_id not in self.edges[gene_id]['children']:
                self.edges[gene_id]['children'].append(cell_id)
            if gene_id not in self.edges[cell_id]['parents']:
                self.edges[cell_id]['parents'].append(gene_id)
            self.edge_attrs[(gene_id, cell_id)] = {'type': 'composition', 'role': None}
            self._mark_edge_dirty(gene_id, cell_id)
            self._mark_node_dirty(gene_id); self._mark_node_dirty(cell_id)

    def log_event_participation(self, participant_id: str, event_id: str, role: str):
        with self._lock:
            self._ensure_node_edges(participant_id)
            self._ensure_node_edges(event_id)
            # participant -> event with role
            if event_id not in self.edges[participant_id]['children']:
                self.edges[participant_id]['children'].append(event_id)
            if participant_id not in self.edges[event_id]['parents']:
                self.edges[event_id]['parents'].append(participant_id)
            self.edge_attrs[(participant_id, event_id)] = {'type': 'participation', 'role': role}
            self._mark_edge_dirty(participant_id, event_id)
            self._mark_node_dirty(participant_id); self._mark_node_dirty(event_id)

    def log_event_output(self, event_id: str, output_id: str, role: str):
        with self._lock:
            self._ensure_node_edges(event_id)
            self._ensure_node_edges(output_id)
            # event -> output with role
            if output_id not in self.edges[event_id]['children']:
                self.edges[event_id]['children'].append(output_id)
            if event_id not in self.edges[output_id]['parents']:
                self.edges[output_id]['parents'].append(event_id)
            self.edge_attrs[(event_id, output_id)] = {'type': 'output', 'role': role}
            self._mark_edge_dirty(event_id, output_id)
            self._mark_node_dirty(event_id); self._mark_node_dirty(output_id)

    def query_action_effect_with_stats(self, action: str, context_filters: Dict, generation_window: int = 10, decay_rate: float = 0.1) -> Dict:
        """Query the effect of an action with detailed statistics. Now reads from the dictionary backend."""
        with self._lock:
            # Create a snapshot to avoid issues with concurrent modification
            nodes_snapshot = list(self.nodes.items())
            ts_snapshot = dict(self.event_timestamps)

        relevant_effects = []
        weights = []
        current_time = datetime.now().timestamp()

        for node_id, data in nodes_snapshot:
            if data.get('type') != 'event':
                continue
            
            try:
                details_str = data.get('details', '{}')
                details = json.loads(details_str)
            except (json.JSONDecodeError, TypeError):
                continue

            if details.get('action') != action:
                continue

            # Context matching
            if all(details.get(k) == v for k, v in context_filters.items()):
                effect = data.get('effect', 0.0)
                timestamp = ts_snapshot.get(node_id, current_time)
                time_diff = (current_time - timestamp) / 3600
                weight = np.exp(-decay_rate * time_diff)
                relevant_effects.append(effect)
                weights.append(weight)
        
        if not relevant_effects:
            return {'effect': 0.0, 'count': 0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        weighted_mean = np.average(relevant_effects, weights=weights) if sum(weights) > 0 else np.mean(relevant_effects)
        
        return {
            'effect': float(weighted_mean),
            'count': len(relevant_effects),
            'std': float(np.std(relevant_effects)) if len(relevant_effects) > 1 else 0.0,
            'min': float(np.min(relevant_effects)),
            'max': float(np.max(relevant_effects))
        }

    def _prune_graph_if_needed(self):
        """Prunes the graph based on the number of nodes."""
        if self.max_graph_size is None or len(self.nodes) <= self.max_graph_size:
            return
        
        with self._lock:
            target_size = int(self.max_graph_size * self.prune_target_fraction)
            num_to_prune = len(self.nodes) - target_size
            
            # Sort by timestamp to remove the oldest entries
            sorted_by_time = sorted(self.event_timestamps.items(), key=lambda item: item[1])
            
            nodes_to_remove = {node_id for node_id, ts in sorted_by_time[:num_to_prune]}
            
            # Remove nodes and their corresponding edges and timestamps
            for node_id in list(self.nodes.keys()):
                if node_id in nodes_to_remove:
                    self.nodes.pop(node_id, None)
                    self.edges.pop(node_id, None)
                    self.event_timestamps.pop(node_id, None)

            # Clean up dangling edges
            for node_id, edge_data in self.edges.items():
                edge_data['parents'] = [p for p in edge_data['parents'] if p not in nodes_to_remove]
                edge_data['children'] = [c for c in edge_data['children'] if c not in nodes_to_remove]

        print(f"Pruned {len(nodes_to_remove)} nodes to maintain graph size.")

    def _prune_by_generation_if_needed(self, current_generation: int):
        """Prunes nodes older than max_generation_age."""
        if not self.generation_prune_enabled or self.max_generation_age is None:
            return
        if current_generation < self._last_prune_gen + self.prune_check_interval_gens:
            return
            
        with self._lock:
            cutoff = current_generation - self.max_generation_age
            nodes_to_remove = {
                node_id for node_id, data in self.nodes.items()
                if data.get('generation', current_generation) < cutoff
            }
            
            if not nodes_to_remove:
                self._last_prune_gen = current_generation
                return

            # Perform removal (similar to size-based pruning)
            for node_id in list(self.nodes.keys()):
                if node_id in nodes_to_remove:
                    self.nodes.pop(node_id, None)
                    self.edges.pop(node_id, None)
                    self.event_timestamps.pop(node_id, None)
            
            for node_id, edge_data in self.edges.items():
                edge_data['parents'] = [p for p in edge_data['parents'] if p not in nodes_to_remove]
                edge_data['children'] = [c for c in edge_data['children'] if c not in nodes_to_remove]

            self._last_prune_gen = current_generation
        print(f"Pruned {len(nodes_to_remove)} old nodes (gen < {cutoff}).")

    # The following methods are mostly internal or unchanged as they don't depend on the graph backend
    def query_action_effect(self, *args, **kwargs) -> float:
        return self.query_action_effect_with_stats(*args, **kwargs).get('effect', 0.0)
    
    def query_causal_direction(self, action: str, context: Dict[str, Any]) -> Optional[np.ndarray]:
        # This method uses directional_effects and direction_stats, which are already dictionary-based
        # and do not need to be changed. The original implementation is fine.
        try:
            action_id = self._get_action_id(str(action))
            ctx_id = self._get_context_id(context)
            key = (action_id, ctx_id)
            stats = self.direction_stats.get(key)
            if stats and stats.get('count', 0) > 0:
                m = np.asarray(stats['mean'], dtype=np.float32)
                n = float(np.linalg.norm(m))
                if n > 1e-12:
                    return (m / n).astype(np.float32)
            # Fallback to legacy buffers
            context_key = self._make_context_key(context)
            cache_key = (action, context_key)
            if cache_key in self._direction_cache:
                return self._direction_cache[cache_key]
            
            buckets = self.directional_effects.get(action)
            if not buckets or context_key not in buckets:
                return None
            
            samples = buckets[context_key]
            if not samples:
                return None
                
            avg_vec = np.mean(np.stack(samples, axis=0), axis=0)
            norm = np.linalg.norm(avg_vec)
            if norm <= 1e-12:
                return None
                
            direction = (avg_vec / norm).astype(float)
            self._direction_cache[cache_key] = direction
            return direction
        except Exception:
            return None

    # Methods like _make_context_key, _get_action_id, _update_direction_stats, etc., are unchanged.
    # We include them here for completeness.
    def _make_context_key(self, details: Dict[str, Any]) -> tuple:
        island = details.get('island')
        pnn_state = details.get('pnn_state')
        stress_bin = details.get('stress_bin')
        parent_types = details.get('parent_types')
        if isinstance(parent_types, (list, tuple)):
            parent_types = tuple(sorted(parent_types))
        return (island, pnn_state, stress_bin, parent_types)

    def _get_action_id(self, val: str) -> int:
        if val not in self._ctx_maps['action']:
            self._ctx_maps['action'][val] = self._ctx_next_id['action']
            self._ctx_next_id['action'] += 1
        return self._ctx_maps['action'][val]

    def _get_context_id(self, context: Dict[str, Any]) -> int:
        # This logic can be simplified or kept as is.
        # For simplicity, we'll keep the existing encoding logic.
        def _get_generic_ctx_id(key: str, val) -> int:
            if val not in self._ctx_maps[key]:
                self._ctx_maps[key][val] = self._ctx_next_id[key]
                self._ctx_next_id[key] += 1
            return self._ctx_maps[key][val]
        
        island_id = _get_generic_ctx_id('island', context.get('island'))
        pnn_id = _get_generic_ctx_id('pnn_state', context.get('pnn_state'))
        parent_types = context.get('parent_types')
        if isinstance(parent_types, (list, tuple)):
            parent_types = tuple(sorted(parent_types))
        parent_id = _get_generic_ctx_id('parent_types', parent_types)
        
        return int((island_id & 0xFF) << 16 | (pnn_id & 0xFF) << 8 | (parent_id & 0xFF))

    def _get_proj(self, orig_dim: int) -> np.ndarray:
        if orig_dim not in self._proj_cache:
            rng = np.random.default_rng(self._proj_seed + orig_dim)
            P = rng.standard_normal((orig_dim, self.PROJ_DIM)).astype(np.float32) / np.sqrt(self.PROJ_DIM)
            self._proj_cache[orig_dim] = P
        return self._proj_cache[orig_dim]

    def _update_direction_stats(self, action_id: int, ctx_id: int, vec: np.ndarray, effect_val: float):
        key = (action_id, ctx_id)
        P = self._get_proj(vec.size)
        vproj = (vec @ P).astype(np.float32)
        
        if key not in self.direction_stats:
            self.direction_stats[key] = {
                'count': 0,
                'mean': np.zeros(self.PROJ_DIM, dtype=np.float32),
                'rb': np.zeros((self.RING_SIZE, self.PROJ_DIM), dtype=np.float16),
                'rb_idx': 0,
            }
        
        stats = self.direction_stats[key]
        c = stats['count'] + 1
        m = stats['mean']
        stats['mean'] += (vproj - m) / c
        stats['count'] = c
        idx = stats['rb_idx'] % self.RING_SIZE
        stats['rb'][idx, :] = vproj.astype(np.float16)
        stats['rb_idx'] += 1

    # ---------------------
    # Embedding helpers
    # ---------------------
    def _mark_node_dirty(self, nid: str):
        try:
            self._embed_dirty_nodes.add(nid)
        except Exception:
            pass

    def _mark_edge_dirty(self, u: str, v: str):
        try:
            self._embed_dirty_edges.add((u, v))
        except Exception:
            pass

    def _fh(self, s: str) -> int:
        return (hash(s) & 0xFFFFFFFFFFFFFFFF) % int(self.HASH_DIM)

    def _hash_features(self, feats: Dict[str, Any]) -> np.ndarray:
        v = np.zeros(int(self.HASH_DIM), dtype=np.float32)
        for k, val in feats.items():
            if val is None:
                continue
            if isinstance(val, (list, tuple)):
                for x in val:
                    v[self._fh(f"{k}={x}")] += 1.0
            elif isinstance(val, (int, float)):
                v[self._fh(k)] += float(val)
            else:
                v[self._fh(f"{k}={val}")] += 1.0
        return v

    def _project_normalize(self, v_hash: np.ndarray) -> np.ndarray:
        P = self._get_proj(v_hash.size)
        z = (v_hash @ P).astype(np.float32)
        n = float(np.linalg.norm(z))
        if n > 1e-12:
            z /= n
        return z.astype(np.float16)

    def _encode_node_vec(self, node_id: str) -> np.ndarray:
        d = self.nodes.get(node_id, {})
        feats = {
            'type': d.get('type'),
            'island': d.get('island'),
            'pnn_state': d.get('pnn_state'),
            'event_type': d.get('event_type'),
            'parent_types': d.get('parent_types'),
            'generation': float(d.get('generation', 0.0)),
            'fitness': float(d.get('fitness', 0.0)) if d.get('type') == 'cell' else 0.0,
            'effect': float(d.get('effect', 0.0)) if d.get('type') == 'event' else 0.0,
            'age_h': 0.0,
        }
        ts = self.event_timestamps.get(node_id)
        if isinstance(ts, (int, float)):
            feats['age_h'] = max(0.0, (datetime.now().timestamp() - ts) / 3600.0)
        v_hash = self._hash_features(feats)
        return self._project_normalize(v_hash)

    def _encode_edge_vec(self, u: str, v: str) -> np.ndarray:
        a = self.node_vecs_fp16.get(u)
        b = self.node_vecs_fp16.get(v)
        if a is None:
            a = self._encode_node_vec(u); self.node_vecs_fp16[u] = a
        if b is None:
            b = self._encode_node_vec(v); self.node_vecs_fp16[v] = b
        attrs = self.edge_attrs.get((u, v), {})
        bias = self._hash_features({'etype': attrs.get('type'), 'role': attrs.get('role')})
        zbias = self._project_normalize(bias).astype(np.float32)
        z = (a.astype(np.float32) + b.astype(np.float32)) * 0.5 + zbias * 0.1
        n = float(np.linalg.norm(z))
        if n > 1e-12:
            z /= n
        return z.astype(np.float16)

    def _refresh_embeddings(self):
        for nid in list(self._embed_dirty_nodes):
            if nid in self.nodes:
                self.node_vecs_fp16[nid] = self._encode_node_vec(nid)
        self._embed_dirty_nodes.clear()
        for e in list(self._embed_dirty_edges):
            u, v = e
            if u in self.nodes and v in self.nodes:
                self.edge_vecs_fp16[e] = self._encode_edge_vec(u, v)
        self._embed_dirty_edges.clear()

    def get_node_vec(self, node_id: str) -> Optional[np.ndarray]:
        if node_id not in self.node_vecs_fp16:
            if node_id not in self.nodes:
                return None
            self.node_vecs_fp16[node_id] = self._encode_node_vec(node_id)
        return self.node_vecs_fp16.get(node_id)

    def get_edge_vec(self, u: str, v: str) -> Optional[np.ndarray]:
        key = (u, v)
        if key not in self.edge_vecs_fp16:
            if u not in self.nodes or v not in self.nodes:
                return None
            self.edge_vecs_fp16[key] = self._encode_edge_vec(u, v)
        return self.edge_vecs_fp16.get(key)

    def similar_nodes(self, node_id: str, k: int = 10) -> List[tuple]:
        self._refresh_embeddings()
        q = self.get_node_vec(node_id)
        if q is None:
            return []
        qf = q.astype(np.float32)
        out = []
        for nid, vec in self.node_vecs_fp16.items():
            if nid == node_id:
                continue
            v = vec.astype(np.float32)
            sim = float(np.dot(qf, v))
            out.append((nid, sim))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:k]

    # Methods for saving/loading/visualization need to be adapted or removed
    def save_tapestry(self, filepath: str):
        print("Warning: save_tapestry with dictionary backend is not fully supported. Exporting to JSON instead.")
        self.export_to_json(filepath.replace('.graphml', '.json'))

    def export_to_json(self, filepath: str, generation_window: Optional[int] = None):
        with self._lock:
            # Reconstruct a temporary graph-like structure for export
            nodes_to_export = []
            links_to_export = []
            
            min_gen = -1
            if generation_window is not None:
                min_gen = self._max_seen_generation - generation_window

            for node_id, data in self.nodes.items():
                if data.get('generation', 0) >= min_gen:
                    nodes_to_export.append({'id': node_id, **data})
                    if node_id in self.edges:
                        for child_id in self.edges[node_id].get('children', []):
                            if child_id in self.nodes and self.nodes[child_id].get('generation', 0) >= min_gen:
                                links_to_export.append({'source': node_id, 'target': child_id})

            export_data = {'nodes': nodes_to_export, 'links': links_to_export}

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Causal Tapestry (dictionary backend) exported to JSON at {filepath}")

    def visualize_tapestry(self, output_path: str, generation_window: int = 10):
        print("Warning: visualize_tapestry is not supported with the high-performance dictionary backend.")

    # Vector save/load separate from JSON
    def save_vectors(self, npz_path: str):
        self._refresh_embeddings()
        node_ids = list(self.node_vecs_fp16.keys())
        if node_ids:
            nvecs = np.stack([self.node_vecs_fp16[i] for i in node_ids], axis=0)
        else:
            nvecs = np.zeros((0, self.PROJ_DIM), dtype=np.float16)
        edge_keys = list(self.edge_vecs_fp16.keys())
        if edge_keys:
            evecs = np.stack([self.edge_vecs_fp16[k] for k in edge_keys], axis=0)
        else:
            evecs = np.zeros((0, self.PROJ_DIM), dtype=np.float16)
        meta = np.array([self.EMBED_VERSION, str(self.PROJ_DIM), str(self.HASH_DIM)], dtype=object)
        np.savez_compressed(
            npz_path,
            node_ids=np.array(node_ids),
            node_vecs=nvecs,
            edge_keys=np.array([f"{u}\t{v}" for (u, v) in edge_keys]),
            edge_vecs=evecs,
            meta=meta,
        )
        print(f"Saved vectors -> {npz_path} ({nvecs.shape[0]} nodes, {evecs.shape[0]} edges)")

    def load_vectors(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.node_vecs_fp16.clear(); self.edge_vecs_fp16.clear()
        for nid, vec in zip(data['node_ids'], data['node_vecs']):
            self.node_vecs_fp16[str(nid)] = vec.astype(np.float16)
        for ek, vec in zip(data['edge_keys'], data['edge_vecs']):
            u, v = str(ek).split('\t', 1)
            self.edge_vecs_fp16[(u, v)] = vec.astype(np.float16)
        self._embed_dirty_nodes.clear(); self._embed_dirty_edges.clear()
        print(f"Loaded vectors <- {npz_path}")

    def load_from_json(self, filepath: str):
        """Load tapestry (nodes/edges) from a JSON export created by export_to_json."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        nodes_in = data.get('nodes', [])
        links_in = data.get('links', [])
        with self._lock:
            self.nodes.clear()
            self.edges.clear()
            self.event_timestamps.clear()
            self._max_seen_generation = 0
            for n in nodes_in:
                nid = str(n.get('id'))
                nd = dict(n)
                nd.pop('id', None)
                self.nodes[nid] = nd
                self._ensure_node_edges(nid)
                gen = int(nd.get('generation', 0) or 0)
                self._max_seen_generation = max(self._max_seen_generation, gen)
                if nd.get('type') == 'event' and 'timestamp' in nd:
                    try:
                        self.event_timestamps[nid] = float(nd['timestamp'])
                    except Exception:
                        pass
            for e in links_in:
                u = str(e.get('source'))
                v = str(e.get('target'))
                self._ensure_node_edges(u)
                self._ensure_node_edges(v)
                if v not in self.edges[u]['children']:
                    self.edges[u]['children'].append(v)
                if u not in self.edges[v]['parents']:
                    self.edges[v]['parents'].append(u)
        # Mark everything dirty so vectors can be re-derived lazily
        try:
            self._embed_dirty_nodes = set(self.nodes.keys())
            self._embed_dirty_edges = set((u, v) for u, d in self.edges.items() for v in d.get('children', []))
        except Exception:
            pass
        print(f"Causal Tapestry loaded from JSON: {filepath} (nodes={len(self.nodes)}, edges~={sum(len(d.get('children', [])) for d in self.edges.values())})")