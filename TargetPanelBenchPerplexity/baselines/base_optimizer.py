"""
Base optimizer class for TargetPanelBench framework.
Provides common interface for all target prioritization algorithms.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging

class BaseTargetOptimizer(ABC):
    """
    Abstract base class for target prioritization algorithms.

    All optimization methods in TargetPanelBench should inherit from this class
    and implement the required abstract methods.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the optimizer.

        Args:
            name: Human-readable name for the optimizer
            **kwargs: Algorithm-specific parameters
        """
        self.name = name
        self.params = kwargs
        self.logger = logging.getLogger(f"TargetPanelBench.{name}")
        self.is_fitted = False

    @abstractmethod
    def fit(self, 
            targets: List[str], 
            evidence_matrix: pd.DataFrame,
            ground_truth: Optional[List[str]] = None) -> 'BaseTargetOptimizer':
        """
        Fit the optimizer to the training data.

        Args:
            targets: List of target gene/protein identifiers
            evidence_matrix: DataFrame with targets as rows and evidence types as columns
            ground_truth: Optional list of known positive targets for training

        Returns:
            self: The fitted optimizer instance
        """
        pass

    @abstractmethod
    def rank_targets(self, 
                    targets: List[str], 
                    evidence_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Rank targets based on prioritization score.

        Args:
            targets: List of target identifiers to rank
            evidence_matrix: Evidence scores for each target

        Returns:
            DataFrame with columns: ['target', 'score', 'rank']
        """
        pass

    @abstractmethod
    def select_panel(self, 
                    ranked_targets: pd.DataFrame,
                    panel_size: int = 15,
                    diversity_weight: float = 0.3) -> List[str]:
        """
        Select diverse panel from ranked targets.

        Args:
            ranked_targets: Output from rank_targets()
            panel_size: Number of targets to select
            diversity_weight: Weight for diversity vs. score (0=pure score, 1=pure diversity)

        Returns:
            List of selected target identifiers
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        return self.params.copy()

    def set_params(self, **params) -> 'BaseTargetOptimizer':
        """Set algorithm parameters."""
        self.params.update(params)
        self.is_fitted = False
        return self

    def __str__(self) -> str:
        return f"{self.name}({self.params})"

    def __repr__(self) -> str:
        return self.__str__()


class EvidenceProcessor:
    """
    Utility class for processing evidence matrices.
    Handles normalization, missing values, and feature engineering.
    """

    @staticmethod
    def normalize_evidence(evidence_df: pd.DataFrame, 
                          method: str = 'min_max') -> pd.DataFrame:
        """
        Normalize evidence scores to [0, 1] range.

        Args:
            evidence_df: Raw evidence matrix
            method: Normalization method ('min_max', 'z_score', 'robust')

        Returns:
            Normalized evidence matrix
        """
        df_normalized = evidence_df.copy()

        for column in df_normalized.select_dtypes(include=[np.number]).columns:
            if method == 'min_max':
                min_val = df_normalized[column].min()
                max_val = df_normalized[column].max()
                if max_val > min_val:
                    df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
                else:
                    df_normalized[column] = 0.0
            elif method == 'z_score':
                mean_val = df_normalized[column].mean()
                std_val = df_normalized[column].std()
                if std_val > 0:
                    df_normalized[column] = (df_normalized[column] - mean_val) / std_val
                else:
                    df_normalized[column] = 0.0
            elif method == 'robust':
                median_val = df_normalized[column].median()
                mad_val = np.median(np.abs(df_normalized[column] - median_val))
                if mad_val > 0:
                    df_normalized[column] = (df_normalized[column] - median_val) / mad_val
                else:
                    df_normalized[column] = 0.0

        return df_normalized

    @staticmethod
    def handle_missing_values(evidence_df: pd.DataFrame, 
                             strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in evidence matrix.

        Args:
            evidence_df: Evidence matrix with potential missing values
            strategy: Strategy for imputation ('median', 'mean', 'zero', 'forward_fill')

        Returns:
            Evidence matrix with imputed values
        """
        df_imputed = evidence_df.copy()

        for column in df_imputed.select_dtypes(include=[np.number]).columns:
            if strategy == 'median':
                df_imputed[column].fillna(df_imputed[column].median(), inplace=True)
            elif strategy == 'mean':
                df_imputed[column].fillna(df_imputed[column].mean(), inplace=True)
            elif strategy == 'zero':
                df_imputed[column].fillna(0.0, inplace=True)
            elif strategy == 'forward_fill':
                df_imputed[column].fillna(method='ffill', inplace=True)
                df_imputed[column].fillna(0.0, inplace=True)  # Handle any remaining NaNs

        return df_imputed
