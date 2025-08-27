"""
Data processing module for TargetPanelBench.

Handles data cleaning, preprocessing, feature engineering, and quality control
for benchmark datasets downloaded from public sources.
"""
import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
import networkx as nx


class DataProcessor:
    """
    Main data processing class for TargetPanelBench.

    Handles cleaning, preprocessing, and feature engineering of raw benchmark data
    to create high-quality datasets for method evaluation.
    """

    def __init__(self, 
                 raw_data_dir: str = "data/cache",
                 processed_data_dir: str = "data/processed",
                 quality_threshold: float = 0.8,
                 min_evidence_sources: int = 3):
        """
        Initialize data processor.

        Args:
            raw_data_dir: Directory containing raw downloaded data
            processed_data_dir: Directory for saving processed data
            quality_threshold: Minimum data quality score to retain targets
            min_evidence_sources: Minimum number of evidence sources per target
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.quality_threshold = quality_threshold
        self.min_evidence_sources = min_evidence_sources

        self.logger = logging.getLogger("TargetPanelBench.DataProcessor")

        # Processing statistics
        self.processing_stats = {
            'targets_before_cleaning': 0,
            'targets_after_cleaning': 0,
            'features_before_cleaning': 0,
            'features_after_cleaning': 0,
            'outliers_removed': 0,
            'missing_values_imputed': 0,
            'duplicates_removed': 0
        }

        # Feature engineering configuration
        self.feature_config = {
            'create_composite_scores': True,
            'add_network_features': True,
            'normalize_features': True,
            'handle_missing_values': True,
            'remove_low_variance': True,
            'detect_outliers': True
        }

    def process_all_data(self, 
                        target_disease_file: str = "opentargets_associations_EFO_0000249.csv",
                        tractability_file: str = "chembl_tractability.csv",
                        ppi_file: str = "string_ppi_network.csv") -> Dict[str, pd.DataFrame]:
        """
        Process all benchmark datasets.

        Args:
            target_disease_file: Target-disease associations filename
            tractability_file: ChEMBL tractability data filename  
            ppi_file: STRING PPI network filename

        Returns:
            Dict of processed datasets
        """
        self.logger.info("Starting comprehensive data processing...")

        # Load raw data
        raw_datasets = self._load_raw_data(target_disease_file, tractability_file, ppi_file)

        # Process each dataset
        processed_datasets = {}

        # 1. Process target-disease associations
        if 'target_disease_associations' in raw_datasets:
            self.logger.info("Processing target-disease associations...")
            processed_datasets['target_disease_associations'] = self._process_target_associations(
                raw_datasets['target_disease_associations']
            )

        # 2. Process tractability data
        if 'tractability' in raw_datasets:
            self.logger.info("Processing tractability data...")
            processed_datasets['tractability'] = self._process_tractability_data(
                raw_datasets['tractability']
            )

        # 3. Process PPI network
        if 'ppi_network' in raw_datasets:
            self.logger.info("Processing PPI network...")
            processed_datasets['ppi_network'] = self._process_ppi_network(
                raw_datasets['ppi_network']
            )

        # 4. Create comprehensive evidence matrix
        self.logger.info("Creating evidence matrix...")
        processed_datasets['evidence_matrix'] = self._create_evidence_matrix(processed_datasets)

        # 5. Generate ground truth set
        self.logger.info("Generating ground truth set...")
        processed_datasets['ground_truth'] = self._generate_ground_truth(
            processed_datasets['target_disease_associations']
        )

        # 6. Create target metadata
        self.logger.info("Creating target metadata...")
        processed_datasets['target_metadata'] = self._create_target_metadata(processed_datasets)

        # 7. Generate quality report
        quality_report = self._generate_quality_report(processed_datasets)
        processed_datasets['quality_report'] = quality_report

        self.logger.info("Data processing complete!")
        self._log_processing_summary()

        return processed_datasets

    def _load_raw_data(self, target_file: str, tractability_file: str, ppi_file: str) -> Dict[str, pd.DataFrame]:
        """Load raw downloaded data files."""
        datasets = {}

        # Load target-disease associations
        target_path = self.raw_data_dir / target_file
        if target_path.exists():
            datasets['target_disease_associations'] = pd.read_csv(target_path)
            self.logger.info(f"Loaded {len(datasets['target_disease_associations'])} target associations")
        else:
            self.logger.warning(f"Target associations file not found: {target_path}")

        # Load tractability data
        tractability_path = self.raw_data_dir / tractability_file
        if tractability_path.exists():
            datasets['tractability'] = pd.read_csv(tractability_path)
            self.logger.info(f"Loaded {len(datasets['tractability'])} tractability records")
        else:
            self.logger.warning(f"Tractability file not found: {tractability_path}")

        # Load PPI network
        ppi_path = self.raw_data_dir / ppi_file
        if ppi_path.exists():
            datasets['ppi_network'] = pd.read_csv(ppi_path)
            self.logger.info(f"Loaded {len(datasets['ppi_network'])} PPI interactions")
        else:
            self.logger.warning(f"PPI network file not found: {ppi_path}")

        return datasets

    def _process_target_associations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean target-disease associations.

        Args:
            df: Raw target associations DataFrame

        Returns:
            Cleaned associations DataFrame
        """
        processed_df = df.copy()
        self.processing_stats['targets_before_cleaning'] = len(processed_df)

        # Remove duplicates
        initial_size = len(processed_df)
        processed_df = processed_df.drop_duplicates(subset=['target_symbol'])
        duplicates_removed = initial_size - len(processed_df)
        self.processing_stats['duplicates_removed'] += duplicates_removed

        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate targets")

        # Clean target symbols (remove invalid characters, standardize format)
        processed_df['target_symbol'] = processed_df['target_symbol'].str.strip().str.upper()
        processed_df = processed_df[processed_df['target_symbol'].str.len() > 1]
        processed_df = processed_df[~processed_df['target_symbol'].str.contains(r'[^A-Z0-9\-_]', regex=True)]

        # Validate and clean evidence scores
        evidence_columns = [
            'overall_score', 'genetic_association', 'somatic_mutation', 
            'known_drug', 'affected_pathway', 'literature', 'rna_expression', 'animal_model'
        ]

        for col in evidence_columns:
            if col in processed_df.columns:
                # Ensure scores are numeric and in [0,1] range
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                processed_df[col] = processed_df[col].clip(0, 1)

        # Remove targets with insufficient evidence
        min_score_threshold = 0.01
        processed_df = processed_df[processed_df['overall_score'] >= min_score_threshold]

        # Count non-zero evidence sources per target
        evidence_counts = (processed_df[evidence_columns[1:]] > 0.01).sum(axis=1)
        processed_df = processed_df[evidence_counts >= self.min_evidence_sources]

        # Handle missing target names and biotypes
        processed_df['target_name'] = processed_df['target_name'].fillna(processed_df['target_symbol'])
        processed_df['biotype'] = processed_df['biotype'].fillna('protein_coding')

        # Sort by overall score (descending)
        processed_df = processed_df.sort_values('overall_score', ascending=False).reset_index(drop=True)

        self.processing_stats['targets_after_cleaning'] = len(processed_df)

        self.logger.info(f"Target associations: {self.processing_stats['targets_before_cleaning']} → {len(processed_df)} targets")

        return processed_df

    def _process_tractability_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean tractability data.

        Args:
            df: Raw tractability DataFrame

        Returns:
            Cleaned tractability DataFrame
        """
        if df.empty:
            # Create empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'target_symbol', 'chembl_id', 'target_type', 'druggability_score',
                'small_molecule_tractable', 'antibody_tractable', 'enzyme_class'
            ])

        processed_df = df.copy()

        # Clean target symbols
        processed_df['target_symbol'] = processed_df['target_symbol'].str.strip().str.upper()

        # Remove duplicates (keep highest druggability score)
        processed_df = processed_df.sort_values('druggability_score', ascending=False)
        processed_df = processed_df.drop_duplicates(subset=['target_symbol'], keep='first')

        # Validate druggability scores
        processed_df['druggability_score'] = pd.to_numeric(processed_df['druggability_score'], errors='coerce')
        processed_df['druggability_score'] = processed_df['druggability_score'].fillna(0.5)
        processed_df['druggability_score'] = processed_df['druggability_score'].clip(0, 1)

        # Clean categorical fields
        processed_df['target_type'] = processed_df['target_type'].fillna('UNKNOWN')
        processed_df['enzyme_class'] = processed_df['enzyme_class'].fillna('UNKNOWN')

        # Standardize boolean fields
        for bool_col in ['small_molecule_tractable', 'antibody_tractable']:
            if bool_col in processed_df.columns:
                processed_df[bool_col] = processed_df[bool_col].astype(bool)

        self.logger.info(f"Processed tractability data for {len(processed_df)} targets")

        return processed_df

    def _process_ppi_network(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean PPI network data.

        Args:
            df: Raw PPI network DataFrame

        Returns:
            Cleaned PPI network DataFrame
        """
        if df.empty:
            return pd.DataFrame(columns=['protein1', 'protein2', 'score'])

        processed_df = df.copy()

        # Clean protein identifiers
        processed_df['protein1'] = processed_df['protein1'].str.strip().str.upper()
        processed_df['protein2'] = processed_df['protein2'].str.strip().str.upper()

        # Remove self-interactions
        processed_df = processed_df[processed_df['protein1'] != processed_df['protein2']]

        # Validate interaction scores
        processed_df['score'] = pd.to_numeric(processed_df['score'], errors='coerce')
        processed_df = processed_df.dropna(subset=['score'])

        # Remove low-confidence interactions
        min_score = 0.4  # STRING medium confidence
        processed_df = processed_df[processed_df['score'] >= min_score]

        # Normalize scores to [0,1]
        if len(processed_df) > 0:
            max_score = processed_df['score'].max()
            if max_score > 1:
                processed_df['score'] = processed_df['score'] / max_score

        # Remove duplicate edges (undirected network)
        # Create sorted pairs to identify duplicates
        processed_df['pair'] = processed_df.apply(
            lambda x: tuple(sorted([x['protein1'], x['protein2']])), axis=1
        )
        processed_df = processed_df.drop_duplicates(subset=['pair'], keep='first')
        processed_df = processed_df.drop('pair', axis=1)

        # Sort by interaction score (descending)
        processed_df = processed_df.sort_values('score', ascending=False).reset_index(drop=True)

        self.logger.info(f"Processed PPI network: {len(processed_df)} high-confidence interactions")

        return processed_df

    def _create_evidence_matrix(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create comprehensive evidence matrix combining all data sources.

        Args:
            datasets: Dict of processed datasets

        Returns:
            Evidence matrix with targets as rows, features as columns
        """
        associations_df = datasets.get('target_disease_associations', pd.DataFrame())
        tractability_df = datasets.get('tractability', pd.DataFrame())
        ppi_df = datasets.get('ppi_network', pd.DataFrame())

        if associations_df.empty:
            raise ValueError("Target-disease associations required for evidence matrix")

        # Start with target associations
        evidence_matrix = associations_df.set_index('target_symbol')[[
            'overall_score', 'genetic_association', 'somatic_mutation',
            'known_drug', 'affected_pathway', 'literature', 'rna_expression', 'animal_model'
        ]].copy()

        self.processing_stats['features_before_cleaning'] = len(evidence_matrix.columns)

        # Add tractability features
        if not tractability_df.empty:
            tractability_features = tractability_df.set_index('target_symbol')

            # Add druggability score
            evidence_matrix['druggability_score'] = tractability_features['druggability_score']

            # Add tractability flags as numeric features
            if 'small_molecule_tractable' in tractability_features.columns:
                evidence_matrix['small_molecule_tractable'] = tractability_features['small_molecule_tractable'].astype(float)

            if 'antibody_tractable' in tractability_features.columns:
                evidence_matrix['antibody_tractable'] = tractability_features['antibody_tractable'].astype(float)

        # Add network-based features
        if not ppi_df.empty and self.feature_config['add_network_features']:
            network_features = self._compute_network_features(ppi_df, evidence_matrix.index.tolist())
            for feature_name, values in network_features.items():
                evidence_matrix[feature_name] = values

        # Add composite features
        if self.feature_config['create_composite_scores']:
            evidence_matrix = self._add_composite_features(evidence_matrix)

        # Handle missing values
        if self.feature_config['handle_missing_values']:
            evidence_matrix = self._handle_missing_values(evidence_matrix)

        # Detect and handle outliers
        if self.feature_config['detect_outliers']:
            evidence_matrix = self._handle_outliers(evidence_matrix)

        # Remove low-variance features
        if self.feature_config['remove_low_variance']:
            evidence_matrix = self._remove_low_variance_features(evidence_matrix)

        # Normalize features
        if self.feature_config['normalize_features']:
            evidence_matrix = self._normalize_features(evidence_matrix)

        self.processing_stats['features_after_cleaning'] = len(evidence_matrix.columns)

        self.logger.info(f"Created evidence matrix: {len(evidence_matrix)} targets × {len(evidence_matrix.columns)} features")

        return evidence_matrix

    def _compute_network_features(self, ppi_df: pd.DataFrame, target_list: List[str]) -> Dict[str, pd.Series]:
        """
        Compute network-based features from PPI data.

        Args:
            ppi_df: PPI network DataFrame
            target_list: List of target identifiers

        Returns:
            Dict mapping feature names to Series of values
        """
        network_features = {}

        try:
            # Create network graph
            G = nx.Graph()

            for _, row in ppi_df.iterrows():
                if row['protein1'] in target_list and row['protein2'] in target_list:
                    G.add_edge(row['protein1'], row['protein2'], weight=row['score'])

            self.logger.info(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            network_features['degree_centrality'] = pd.Series(
                [degree_centrality.get(target, 0.0) for target in target_list], 
                index=target_list
            )

            # Betweenness centrality  
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
            network_features['betweenness_centrality'] = pd.Series(
                [betweenness_centrality.get(target, 0.0) for target in target_list],
                index=target_list
            )

            # Clustering coefficient
            clustering = nx.clustering(G)
            network_features['clustering_coefficient'] = pd.Series(
                [clustering.get(target, 0.0) for target in target_list],
                index=target_list
            )

            # PageRank
            pagerank = nx.pagerank(G, max_iter=100)
            network_features['pagerank'] = pd.Series(
                [pagerank.get(target, 1.0/G.number_of_nodes() if G.number_of_nodes() > 0 else 0.0) for target in target_list],
                index=target_list
            )

            # Eigenvector centrality (with fallback for disconnected components)
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
                network_features['eigenvector_centrality'] = pd.Series(
                    [eigenvector_centrality.get(target, 0.0) for target in target_list],
                    index=target_list
                )
            except nx.NetworkXError:
                # Fallback for disconnected graphs
                network_features['eigenvector_centrality'] = pd.Series(0.0, index=target_list)

        except Exception as e:
            self.logger.warning(f"Network feature computation failed: {e}")
            # Fallback: create default network features
            for feature_name in ['degree_centrality', 'betweenness_centrality', 'clustering_coefficient', 'pagerank', 'eigenvector_centrality']:
                network_features[feature_name] = pd.Series(0.1, index=target_list)

        return network_features

    def _add_composite_features(self, evidence_matrix: pd.DataFrame) -> pd.DataFrame:
        """Add composite features derived from existing evidence."""
        df = evidence_matrix.copy()

        # Evidence diversity score (number of non-zero evidence types)
        evidence_cols = ['genetic_association', 'known_drug', 'literature', 'rna_expression', 'affected_pathway']
        available_evidence_cols = [col for col in evidence_cols if col in df.columns]
        df['evidence_diversity'] = (df[available_evidence_cols] > 0.01).sum(axis=1) / len(available_evidence_cols)

        # Weighted evidence score (emphasizing genetics and known drugs)
        weights = {
            'genetic_association': 0.3,
            'known_drug': 0.25,
            'literature': 0.15,
            'rna_expression': 0.15,
            'affected_pathway': 0.15
        }

        weighted_score = 0
        total_weight = 0
        for col, weight in weights.items():
            if col in df.columns:
                weighted_score += weight * df[col]
                total_weight += weight

        if total_weight > 0:
            df['weighted_evidence_score'] = weighted_score / total_weight

        # Target attractiveness score (combining druggability and evidence)
        if 'druggability_score' in df.columns:
            df['target_attractiveness'] = (
                0.6 * df['overall_score'] + 
                0.4 * df['druggability_score']
            )

        # Network importance score (if network features available)
        network_cols = ['degree_centrality', 'betweenness_centrality', 'pagerank']
        available_network_cols = [col for col in network_cols if col in df.columns]

        if available_network_cols:
            df['network_importance'] = df[available_network_cols].mean(axis=1)

        self.logger.info(f"Added {4 if 'druggability_score' in df.columns else 3} composite features")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in evidence matrix."""
        df_processed = df.copy()

        # Count missing values
        missing_counts = df_processed.isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            self.logger.info(f"Handling {total_missing} missing values...")

            # Use median imputation for most features
            imputer = SimpleImputer(strategy='median')
            df_processed[df_processed.columns] = imputer.fit_transform(df_processed)

            self.processing_stats['missing_values_imputed'] = total_missing

        return df_processed

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in evidence matrix."""
        df_processed = df.copy()
        outliers_removed = 0

        for column in df_processed.columns:
            if df_processed[column].dtype in ['float64', 'int64']:
                # Use IQR method for outlier detection
                Q1 = df_processed[column].quantile(0.25)
                Q3 = df_processed[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Count outliers
                outliers = (df_processed[column] < lower_bound) | (df_processed[column] > upper_bound)
                column_outliers = outliers.sum()

                if column_outliers > 0:
                    # Cap outliers rather than remove them (preserve data)
                    df_processed[column] = df_processed[column].clip(lower_bound, upper_bound)
                    outliers_removed += column_outliers

        if outliers_removed > 0:
            self.processing_stats['outliers_removed'] = outliers_removed
            self.logger.info(f"Capped {outliers_removed} outlier values")

        return df_processed

    def _remove_low_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with very low variance."""
        df_processed = df.copy()

        # Remove features with variance below threshold
        selector = VarianceThreshold(threshold=0.01)

        try:
            features_mask = selector.fit_transform(df_processed)
            selected_features = df_processed.columns[selector.get_support()]

            removed_features = len(df_processed.columns) - len(selected_features)
            if removed_features > 0:
                self.logger.info(f"Removed {removed_features} low-variance features")
                df_processed = df_processed[selected_features]

        except Exception as e:
            self.logger.warning(f"Low variance feature removal failed: {e}")

        return df_processed

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [0,1] range."""
        df_processed = df.copy()

        # Use Min-Max scaling for interpretability
        scaler = MinMaxScaler()

        try:
            df_processed[df_processed.columns] = scaler.fit_transform(df_processed)
            self.logger.info("Applied Min-Max normalization to all features")

        except Exception as e:
            self.logger.warning(f"Feature normalization failed: {e}")

        return df_processed

    def _generate_ground_truth(self, associations_df: pd.DataFrame) -> List[str]:
        """
        Generate ground truth set of validated targets.

        Args:
            associations_df: Processed target associations

        Returns:
            List of ground truth target symbols
        """
        if associations_df.empty:
            return []

        # Define ground truth criteria
        ground_truth_targets = associations_df[
            (associations_df['genetic_association'] >= 0.7) |
            (associations_df['known_drug'] >= 0.8) |
            (associations_df['overall_score'] >= 0.8)
        ]['target_symbol'].tolist()

        # Ensure minimum size
        if len(ground_truth_targets) < 10:
            # Fall back to top-scoring targets
            top_targets = associations_df.nlargest(
                max(15, len(ground_truth_targets)), 'overall_score'
            )['target_symbol'].tolist()
            ground_truth_targets = list(set(ground_truth_targets) | set(top_targets))

        # Limit to reasonable size
        ground_truth_targets = ground_truth_targets[:20]

        self.logger.info(f"Generated ground truth set: {len(ground_truth_targets)} targets")

        return ground_truth_targets

    def _create_target_metadata(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create comprehensive target metadata."""
        associations_df = datasets.get('target_disease_associations', pd.DataFrame())

        if associations_df.empty:
            return pd.DataFrame()

        metadata = associations_df[['target_symbol', 'target_name', 'biotype', 'target_id']].copy()

        # Add data source availability flags
        tractability_df = datasets.get('tractability', pd.DataFrame())
        if not tractability_df.empty:
            metadata['has_tractability_data'] = metadata['target_symbol'].isin(
                tractability_df['target_symbol']
            )

        ppi_df = datasets.get('ppi_network', pd.DataFrame())
        if not ppi_df.empty:
            ppi_targets = set(ppi_df['protein1'].tolist() + ppi_df['protein2'].tolist())
            metadata['has_ppi_data'] = metadata['target_symbol'].isin(ppi_targets)

        # Add ground truth labels
        ground_truth = datasets.get('ground_truth', [])
        metadata['is_ground_truth'] = metadata['target_symbol'].isin(ground_truth)

        # Add data quality scores
        evidence_matrix = datasets.get('evidence_matrix', pd.DataFrame())
        if not evidence_matrix.empty:
            # Calculate completeness score (fraction of non-zero evidence)
            completeness_scores = (evidence_matrix > 0.01).mean(axis=1)
            metadata['data_completeness'] = metadata['target_symbol'].map(completeness_scores)

            # Calculate evidence strength (mean of non-zero evidence)
            evidence_strength = evidence_matrix.apply(
                lambda row: row[row > 0.01].mean() if (row > 0.01).any() else 0.0, axis=1
            )
            metadata['evidence_strength'] = metadata['target_symbol'].map(evidence_strength)

        metadata = metadata.set_index('target_symbol')

        self.logger.info(f"Created metadata for {len(metadata)} targets")

        return metadata

    def _generate_quality_report(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        report = {
            'processing_stats': self.processing_stats.copy(),
            'dataset_sizes': {},
            'data_quality_scores': {},
            'feature_statistics': {},
            'recommendations': []
        }

        # Dataset sizes
        for name, dataset in datasets.items():
            if isinstance(dataset, pd.DataFrame):
                report['dataset_sizes'][name] = {
                    'rows': len(dataset),
                    'columns': len(dataset.columns) if hasattr(dataset, 'columns') else 0
                }
            elif isinstance(dataset, list):
                report['dataset_sizes'][name] = {'items': len(dataset)}

        # Evidence matrix quality analysis
        if 'evidence_matrix' in datasets:
            evidence_matrix = datasets['evidence_matrix']

            # Feature statistics
            report['feature_statistics'] = {
                'mean_values': evidence_matrix.mean().to_dict(),
                'std_values': evidence_matrix.std().to_dict(),
                'min_values': evidence_matrix.min().to_dict(),
                'max_values': evidence_matrix.max().to_dict(),
                'missing_percentages': (evidence_matrix.isnull().mean() * 100).to_dict()
            }

            # Overall quality score
            completeness = 1 - evidence_matrix.isnull().mean().mean()
            variance_score = min(1.0, evidence_matrix.std().mean())
            coverage_score = len(evidence_matrix) / max(500, len(evidence_matrix))  # Target 500+ targets

            overall_quality = (completeness + variance_score + coverage_score) / 3
            report['data_quality_scores']['overall_quality'] = overall_quality
            report['data_quality_scores']['completeness'] = completeness
            report['data_quality_scores']['variance'] = variance_score
            report['data_quality_scores']['coverage'] = coverage_score

        # Generate recommendations
        recommendations = []

        if report['data_quality_scores'].get('completeness', 1.0) < 0.9:
            recommendations.append("Consider additional data sources to improve completeness")

        if report['data_quality_scores'].get('coverage', 1.0) < 0.8:
            recommendations.append("Increase target universe size for more robust benchmarking")

        if len(datasets.get('ground_truth', [])) < 15:
            recommendations.append("Expand ground truth set for more reliable evaluation")

        if 'ppi_network' in datasets and len(datasets['ppi_network']) < 1000:
            recommendations.append("Consider additional PPI data sources for better network features")

        report['recommendations'] = recommendations

        return report

    def _log_processing_summary(self):
        """Log summary of processing statistics."""
        stats = self.processing_stats

        self.logger.info("Data Processing Summary:")
        self.logger.info(f"  Targets: {stats['targets_before_cleaning']} → {stats['targets_after_cleaning']}")
        self.logger.info(f"  Features: {stats['features_before_cleaning']} → {stats['features_after_cleaning']}")
        self.logger.info(f"  Duplicates removed: {stats['duplicates_removed']}")
        self.logger.info(f"  Missing values imputed: {stats['missing_values_imputed']}")
        self.logger.info(f"  Outliers handled: {stats['outliers_removed']}")

    def save_processed_data(self, datasets: Dict[str, Any], 
                           include_quality_report: bool = True) -> None:
        """
        Save processed datasets to files.

        Args:
            datasets: Dict of processed datasets
            include_quality_report: Whether to save quality report
        """
        for name, data in datasets.items():
            if isinstance(data, pd.DataFrame):
                # Save DataFrames as CSV
                filepath = self.processed_data_dir / f"{name}.csv"

                if name == 'evidence_matrix':
                    # Save with target symbols as index
                    data.to_csv(filepath, index=True)
                else:
                    data.to_csv(filepath, index=False)

                self.logger.info(f"Saved {name}: {len(data)} rows → {filepath}")

            elif isinstance(data, (list, dict)):
                # Save lists and dicts as JSON
                filepath = self.processed_data_dir / f"{name}.json"
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

                self.logger.info(f"Saved {name} → {filepath}")

        self.logger.info(f"All processed data saved to {self.processed_data_dir}")


def main():
    """Main function for data processing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize processor
    processor = DataProcessor(
        raw_data_dir="data/cache",
        processed_data_dir="data/processed",
        quality_threshold=0.8
    )

    # Process all data
    processed_datasets = processor.process_all_data()

    # Save processed datasets
    processor.save_processed_data(processed_datasets)

    # Print quality report summary
    quality_report = processed_datasets.get('quality_report', {})

    print("\n" + "="*50)
    print("DATA PROCESSING COMPLETE")
    print("="*50)

    print(f"\nDataset Sizes:")
    for name, size_info in quality_report.get('dataset_sizes', {}).items():
        if 'rows' in size_info:
            print(f"  {name}: {size_info['rows']} rows × {size_info['columns']} columns")
        elif 'items' in size_info:
            print(f"  {name}: {size_info['items']} items")

    print(f"\nData Quality Scores:")
    quality_scores = quality_report.get('data_quality_scores', {})
    for metric, score in quality_scores.items():
        print(f"  {metric}: {score:.3f}")

    recommendations = quality_report.get('recommendations', [])
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")


if __name__ == "__main__":
    main()
