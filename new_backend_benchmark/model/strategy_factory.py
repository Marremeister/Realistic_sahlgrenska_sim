"""
Factory for accessing optimization strategies for benchmarking.
Provides a unified interface for strategy creation and management.
"""

from typing import Dict, List, Any, Optional
from Model.Assignment_strategies.strategy_registry import STRATEGY_REGISTRY


class StrategyFactory:
    """
    Factory class for creating and accessing optimization strategies.

    Leverages the existing strategy registry to provide unified access to
    all optimization strategies in the system.
    """

    @staticmethod
    def get_strategy(strategy_name: str, **kwargs):
        """
        Get a strategy instance by name.

        Args:
            strategy_name: Name of the strategy (e.g., "ILP: Makespan")
            **kwargs: Additional parameters for the strategy

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy not found
        """
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Get the factory function from the registry
        strategy_factory = STRATEGY_REGISTRY[strategy_name]

        # If it's a lambda/factory function, call it to get an instance
        if callable(strategy_factory):
            return strategy_factory()

        # If it's a class, instantiate it
        return strategy_factory()

    @staticmethod
    def get_available_strategies() -> List[str]:
        """
        Get list of available strategy names.

        Returns:
            List[str]: List of strategy names
        """
        return list(STRATEGY_REGISTRY.keys())

    @staticmethod
    def get_strategy_info() -> List[Dict[str, str]]:
        """
        Get information about all available strategies.

        Returns:
            List[Dict[str, str]]: List with information about each strategy
        """
        descriptions = {
            "ILP: Makespan": "Minimizes the maximum completion time across all transporters",
            "ILP: Equal Workload": "Distributes transport requests evenly among transporters",
            "ILP: Urgency First": "Prioritizes urgent transport requests over non-urgent ones",
            "ILP: Cluster-Based": "Divides the hospital into geographical clusters for more efficient assignments",
            "Genetic Algorithm": "Evolves an optimal solution using genetic algorithm techniques",
            "Random": "Assigns transport requests randomly to available transporters"
        }

        strategy_info = []
        for name in STRATEGY_REGISTRY:
            strategy_info.append({
                "id": name,
                "name": name,
                "description": descriptions.get(name, "Strategy description not available")
            })
        return strategy_info