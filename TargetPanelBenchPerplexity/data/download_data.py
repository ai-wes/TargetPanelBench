"""
Data download module for TargetPanelBench.

Handles downloading and preprocessing of public datasets from:
- Open Targets Platform (target-disease associations)
- ChEMBL (tractability data)
- STRING (protein-protein interactions) 
- Other public sources

All data is public domain and no proprietary information is used.
"""
import pandas as pd
import numpy as np
import requests
import json
import time
import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import gzip
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed


class DataDownloader:
    """
    Main class for downloading benchmark datasets.

    Downloads and caches data from multiple public APIs and databases
    to create a comprehensive target prioritization benchmark.
    """

    def __init__(self, 
                 cache_dir: str = "data/cache",
                 disease_id: str = "EFO_0000249",  # Alzheimer's disease
                 max_targets: int = 1000,
                 min_evidence_score: float = 0.1):
        """
        Initialize data downloader.

        Args:
            cache_dir: Directory for caching downloaded data
            disease_id: Open Targets disease ID (default: Alzheimer's)
            max_targets: Maximum number of targets to include
            min_evidence_score: Minimum evidence score threshold
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.disease_id = disease_id
        self.max_targets = max_targets
        self.min_evidence_score = min_evidence_score

        self.logger = logging.getLogger("TargetPanelBench.DataDownloader")

        # API endpoints
        self.opentargets_api = "https://api.platform.opentargets.org/api/v4"
        self.string_api = "https://string-db.org/api"
        self.chembl_api = "https://www.ebi.ac.uk/chembl/api/data"

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TargetPanelBench/1.0 (Research Use)',
            'Accept': 'application/json'
        })

    def download_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Download all required datasets for the benchmark.

        Returns:
            Dict mapping dataset names to DataFrames
        """
        self.logger.info("Starting benchmark data download...")

        datasets = {}

        # 1. Download target-disease associations from Open Targets
        self.logger.info("Downloading Open Targets associations...")
        datasets['target_disease_associations'] = self.download_opentargets_associations()

        # 2. Download tractability data from ChEMBL
        self.logger.info("Downloading ChEMBL tractability data...")
        target_list = datasets['target_disease_associations']['target_symbol'].tolist()
        datasets['tractability'] = self.download_chembl_tractability(target_list[:self.max_targets])

        # 3. Download protein-protein interactions from STRING
        self.logger.info("Downloading STRING PPI data...")
        datasets['ppi_network'] = self.download_string_interactions(target_list[:self.max_targets])

        # 4. Create evidence matrix
        self.logger.info("Creating evidence matrix...")
        datasets['evidence_matrix'] = self.create_evidence_matrix(datasets)

        # 5. Define ground truth set
        self.logger.info("Creating ground truth set...")
        datasets['ground_truth'] = self.create_ground_truth(datasets['target_disease_associations'])

        self.logger.info(f"Data download complete. Downloaded {len(datasets)} datasets.")
        return datasets

    def download_opentargets_associations(self) -> pd.DataFrame:
        """
        Download target-disease associations from Open Targets Platform.

        Returns:
            DataFrame with target associations and evidence scores
        """
        cache_file = self.cache_dir / f"opentargets_associations_{self.disease_id}.csv"

        if cache_file.exists():
            self.logger.info("Loading cached Open Targets data...")
            return pd.read_csv(cache_file)

        # Query Open Targets GraphQL API
        query = """
        query targetAssociationsQuery($diseaseId: String!, $size: Int!) {
          disease(efoId: $diseaseId) {
            id
            name
            associatedTargets(size: $size, orderBy: {score: desc}) {
              count
              rows {
                target {
                  id
                  approvedSymbol
                  approvedName
                  biotype
                }
                score
                datatypeScores {
                  id
                  score
                }
              }
            }
          }
        }
        """

        variables = {
            "diseaseId": self.disease_id,
            "size": self.max_targets
        }

        response = self._post_graphql_query(query, variables)

        if response is None:
            self.logger.error("Failed to download Open Targets data")
            return pd.DataFrame()

        # Parse response
        associations_data = []
        disease_data = response.get('data', {}).get('disease', {})

        if not disease_data:
            self.logger.error("No disease data found")
            return pd.DataFrame()

        associated_targets = disease_data.get('associatedTargets', {}).get('rows', [])

        for row in associated_targets:
            target = row['target']

            # Extract datatype scores
            datatype_scores = {dt['id']: dt['score'] for dt in row['datatypeScores']}

            associations_data.append({
                'target_id': target['id'],
                'target_symbol': target['approvedSymbol'],
                'target_name': target['approvedName'],
                'biotype': target['biotype'],
                'overall_score': row['score'],
                'genetic_association': datatype_scores.get('genetic_association', 0.0),
                'somatic_mutation': datatype_scores.get('somatic_mutation', 0.0),
                'known_drug': datatype_scores.get('known_drug', 0.0),
                'affected_pathway': datatype_scores.get('affected_pathway', 0.0),
                'literature': datatype_scores.get('literature', 0.0),
                'rna_expression': datatype_scores.get('rna_expression', 0.0),
                'animal_model': datatype_scores.get('animal_model', 0.0)
            })

        df = pd.DataFrame(associations_data)

        # Filter by minimum evidence score
        df = df[df['overall_score'] >= self.min_evidence_score]

        # Cache results
        df.to_csv(cache_file, index=False)
        self.logger.info(f"Downloaded {len(df)} target-disease associations")

        return df

    def download_chembl_tractability(self, target_symbols: List[str]) -> pd.DataFrame:
        """
        Download tractability data from ChEMBL.

        Args:
            target_symbols: List of target gene symbols

        Returns:
            DataFrame with tractability information
        """
        cache_file = self.cache_dir / "chembl_tractability.csv"

        if cache_file.exists():
            self.logger.info("Loading cached ChEMBL tractability data...")
            return pd.read_csv(cache_file)

        tractability_data = []

        # Process targets in batches
        batch_size = 50
        for i in range(0, len(target_symbols), batch_size):
            batch = target_symbols[i:i+batch_size]

            # Query ChEMBL API for target information
            for symbol in batch:
                try:
                    target_info = self._get_chembl_target_info(symbol)
                    if target_info:
                        tractability_data.append(target_info)

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    self.logger.warning(f"Failed to get ChEMBL data for {symbol}: {e}")
                    continue

        df = pd.DataFrame(tractability_data)

        if len(df) > 0:
            # Cache results
            df.to_csv(cache_file, index=False)
            self.logger.info(f"Downloaded tractability data for {len(df)} targets")
        else:
            self.logger.warning("No ChEMBL tractability data downloaded")
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=[
                'target_symbol', 'chembl_id', 'target_type', 'druggability_score',
                'small_molecule_tractable', 'antibody_tractable', 'enzyme_class'
            ])

        return df

    def download_string_interactions(self, target_symbols: List[str]) -> pd.DataFrame:
        """
        Download protein-protein interactions from STRING database.

        Args:
            target_symbols: List of target gene symbols

        Returns:
            DataFrame with PPI network edges
        """
        cache_file = self.cache_dir / "string_ppi_network.csv"

        if cache_file.exists():
            self.logger.info("Loading cached STRING PPI data...")
            return pd.read_csv(cache_file)

        # Convert symbols to STRING identifiers
        string_ids = self._map_symbols_to_string_ids(target_symbols)

        if not string_ids:
            self.logger.warning("No STRING IDs found for targets")
            return pd.DataFrame(columns=['protein1', 'protein2', 'score'])

        # Download network interactions
        interactions_data = []

        # Query STRING API for network
        identifiers = "%0d".join(string_ids)

        params = {
            'identifiers': identifiers,
            'species': 9606,  # Human
            'required_score': 400,  # Medium confidence
            'network_type': 'functional',
            'caller_identity': 'TargetPanelBench'
        }

        url = f"{self.string_api}/tsv/network"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Parse TSV response
            lines = response.text.strip().split('\n')
            if len(lines) > 1:  # Has header
                headers = lines[0].split('\t')
                for line in lines[1:]:
                    values = line.split('\t')
                    if len(values) >= 3:
                        interactions_data.append({
                            'protein1': values[2],  # preferredName_A
                            'protein2': values[3],  # preferredName_B
                            'score': float(values[5]) if len(values) > 5 else 0.0  # combined score
                        })

        except Exception as e:
            self.logger.error(f"Failed to download STRING interactions: {e}")

        df = pd.DataFrame(interactions_data)

        if len(df) > 0:
            # Cache results
            df.to_csv(cache_file, index=False)
            self.logger.info(f"Downloaded {len(df)} protein-protein interactions")
        else:
            self.logger.warning("No STRING interactions downloaded")
            df = pd.DataFrame(columns=['protein1', 'protein2', 'score'])

        return df

    def create_evidence_matrix(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create comprehensive evidence matrix combining all data sources.

        Args:
            datasets: Dict of downloaded datasets

        Returns:
            Evidence matrix with targets as rows, evidence types as columns
        """
        associations_df = datasets['target_disease_associations']
        tractability_df = datasets.get('tractability', pd.DataFrame())
        ppi_df = datasets.get('ppi_network', pd.DataFrame())

        # Start with Open Targets associations
        evidence_matrix = associations_df.set_index('target_symbol')[[
            'overall_score', 'genetic_association', 'known_drug', 
            'literature', 'rna_expression', 'affected_pathway'
        ]].copy()

        # Add tractability scores
        if len(tractability_df) > 0:
            tractability_scores = tractability_df.set_index('target_symbol')['druggability_score']
            evidence_matrix['druggability_score'] = tractability_scores
        else:
            evidence_matrix['druggability_score'] = 0.5  # Default moderate score

        # Add PPI network centrality scores
        if len(ppi_df) > 0:
            centrality_scores = self._calculate_network_centrality(ppi_df, evidence_matrix.index.tolist())
            evidence_matrix['ppi_centrality'] = centrality_scores
        else:
            evidence_matrix['ppi_centrality'] = 0.1  # Default low centrality

        # Add expression specificity score (simplified - would use GTEx data in full implementation)
        evidence_matrix['expression_specificity'] = np.random.uniform(0.1, 0.9, len(evidence_matrix))

        # Fill missing values with median
        evidence_matrix = evidence_matrix.fillna(evidence_matrix.median())

        self.logger.info(f"Created evidence matrix with {len(evidence_matrix)} targets and {len(evidence_matrix.columns)} evidence types")

        return evidence_matrix

    def create_ground_truth(self, associations_df: pd.DataFrame) -> List[str]:
        """
        Create ground truth set of validated targets.

        Args:
            associations_df: Target-disease associations DataFrame

        Returns:
            List of ground truth target symbols
        """
        # Use targets with high genetic evidence or known drugs as ground truth
        ground_truth_targets = associations_df[
            (associations_df['genetic_association'] >= 0.7) |
            (associations_df['known_drug'] >= 0.8) |
            (associations_df['overall_score'] >= 0.8)
        ]['target_symbol'].tolist()

        # Ensure we have at least 10 ground truth targets
        if len(ground_truth_targets) < 10:
            # Fall back to top-scoring targets
            ground_truth_targets = associations_df.nlargest(
                max(10, len(ground_truth_targets)), 'overall_score'
            )['target_symbol'].tolist()

        # Limit to reasonable size
        ground_truth_targets = ground_truth_targets[:min(20, len(ground_truth_targets))]

        self.logger.info(f"Created ground truth set with {len(ground_truth_targets)} targets")

        return ground_truth_targets

    def _post_graphql_query(self, query: str, variables: Dict) -> Optional[Dict]:
        """Send GraphQL query to Open Targets."""
        url = f"{self.opentargets_api}/graphql"

        payload = {
            'query': query,
            'variables': variables
        }

        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"GraphQL query failed: {e}")
            return None

    def _get_chembl_target_info(self, symbol: str) -> Optional[Dict]:
        """Get target information from ChEMBL API."""
        # Search for target by symbol
        search_url = f"{self.chembl_api}/target/search"

        try:
            response = self.session.get(
                search_url, 
                params={'q': symbol, 'format': 'json'},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                targets = data.get('targets', [])

                if targets:
                    target = targets[0]  # Take first match
                    return {
                        'target_symbol': symbol,
                        'chembl_id': target.get('target_chembl_id', ''),
                        'target_type': target.get('target_type', 'UNKNOWN'),
                        'druggability_score': min(1.0, len(target.get('target_components', [])) * 0.2),
                        'small_molecule_tractable': 'SMALL_MOLECULE' in target.get('target_type', ''),
                        'antibody_tractable': target.get('target_type') in ['PROTEIN_COMPLEX', 'SINGLE_PROTEIN'],
                        'enzyme_class': target.get('target_type', 'UNKNOWN')
                    }

        except Exception as e:
            self.logger.debug(f"ChEMBL query failed for {symbol}: {e}")

        return None

    def _map_symbols_to_string_ids(self, symbols: List[str]) -> List[str]:
        """Map gene symbols to STRING database identifiers."""
        # Use STRING API to map identifiers
        identifiers_str = "%0d".join(symbols[:100])  # Limit batch size

        params = {
            'identifiers': identifiers_str,
            'species': 9606,  # Human
            'echo_query': 1,
            'caller_identity': 'TargetPanelBench'
        }

        url = f"{self.string_api}/tsv/get_string_ids"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Parse response
            lines = response.text.strip().split('\n')
            string_ids = []

            if len(lines) > 1:  # Has header
                for line in lines[1:]:
                    values = line.split('\t')
                    if len(values) >= 2:
                        string_ids.append(values[1])  # STRING identifier

            return string_ids

        except Exception as e:
            self.logger.warning(f"Failed to map STRING IDs: {e}")
            return []

    def _calculate_network_centrality(self, ppi_df: pd.DataFrame, target_symbols: List[str]) -> pd.Series:
        """Calculate network centrality scores for targets."""
        try:
            import networkx as nx

            # Create network
            G = nx.Graph()

            for _, row in ppi_df.iterrows():
                if row['protein1'] in target_symbols and row['protein2'] in target_symbols:
                    G.add_edge(row['protein1'], row['protein2'], weight=row['score']/1000.0)

            # Calculate centrality
            if len(G.nodes()) > 0:
                centrality = nx.degree_centrality(G)
            else:
                centrality = {}

            # Create series with all targets
            centrality_series = pd.Series(index=target_symbols, dtype=float)
            for target in target_symbols:
                centrality_series[target] = centrality.get(target, 0.01)  # Default low centrality

            return centrality_series

        except ImportError:
            self.logger.warning("NetworkX not available, using random centrality scores")
            return pd.Series(
                np.random.uniform(0.01, 0.5, len(target_symbols)), 
                index=target_symbols
            )

    def save_processed_data(self, datasets: Dict[str, pd.DataFrame], output_dir: str = "data/processed"):
        """Save processed datasets to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                filepath = output_path / f"{name}.csv"
                df.to_csv(filepath, index=True if name == 'evidence_matrix' else False)
                self.logger.info(f"Saved {name} to {filepath}")
            elif isinstance(df, list):
                filepath = output_path / f"{name}.json"
                with open(filepath, 'w') as f:
                    json.dump(df, f, indent=2)
                self.logger.info(f"Saved {name} to {filepath}")


def main():
    """Main function for data download script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Download data for Alzheimer's disease
    downloader = DataDownloader(
        disease_id="EFO_0000249",  # Alzheimer's disease
        max_targets=500,
        min_evidence_score=0.1
    )

    datasets = downloader.download_all_data()
    downloader.save_processed_data(datasets)

    print("\nData download summary:")
    for name, data in datasets.items():
        if isinstance(data, pd.DataFrame):
            print(f"  {name}: {len(data)} rows, {len(data.columns)} columns")
        elif isinstance(data, list):
            print(f"  {name}: {len(data)} items")


if __name__ == "__main__":
    main()
