import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import json
import logging
from loguru import logger

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class TrainingConfig:
    """Configuration for weight learning."""
    data_path: str = "./output/run_20250516_143206/results/hybrid_search_summary.csv"
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5  # for cross-validation
    output_dir: str = "./output/learned_weights"
    min_alpha: float = 0.0
    max_alpha: float = 1.0
    n_alphas: int = 100  # number of alpha values to try

class WeightLearner:
    """Class to learn optimal weights between symbolic and semantic scores."""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize the weight learner."""
        self.config = config if config else TrainingConfig()
        self.data: Optional[pd.DataFrame] = None
        self.best_alpha: Optional[float] = None
        self.best_score: float = -1.0
        self.train_results: Dict[str, List[float]] = {
            'alphas': [],
            'scores': [],
            'precisions': [],
            'recalls': []
        }
        
    def load_data(self, data_path: Optional[str] = None) -> None:
        """Load and prepare the training data.
        
        Args:
            data_path: Path to the CSV file containing the training data.
        """
        data_path = data_path or self.config.data_path
        logger.info(f"Loading training data from {data_path}")
        
        # Load the data
        self.data = pd.read_csv(data_path)
        
        # Create binary labels: 1 for rank 1 matches, 0 otherwise
        self.data['label'] = (self.data['match_rank'] == 1).astype(int)
        
        logger.info(f"Loaded {len(self.data)} samples with {self.data['label'].sum()} positive examples")
    
    def prepare_features_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features (symbolic and semantic scores) and labels.
        
        Returns:
            Tuple containing:
                - X: Array of shape (n_samples, 2) with symbolic and semantic scores
                - y: Array of shape (n_samples,) with binary labels (1 for match, 0 otherwise)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Use symbolic and semantic scores as features
        X = self.data[['symbolic_score', 'semantic_score']].values
        y = self.data['label'].values
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and test sets.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
    
    def calculate_hybrid_score(self, X: np.ndarray, alpha: float) -> np.ndarray:
        """Calculate hybrid score given features and alpha.
        
        Args:
            X: Feature matrix with columns [symbolic_score, semantic_score]
            alpha: Weight for semantic score (1-alpha for symbolic score)
            
        Returns:
            Array of hybrid scores
        """
        return (1 - alpha) * X[:, 0] + alpha * X[:, 1]
    
    def evaluate_alpha(self, X: np.ndarray, y: np.ndarray, alpha: float) -> float:
        """Evaluate performance for a given alpha using average precision.
        
        Args:
            X: Feature matrix
            y: True labels
            alpha: Weight for semantic score
            
        Returns:
            Average precision score
        """
        y_scores = self.calculate_hybrid_score(X, alpha)
        return average_precision_score(y, y_scores)
    
    def find_optimal_alpha(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Find the optimal alpha that maximizes average precision on the training set.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Optimal alpha value
        """
        alphas = np.linspace(
            self.config.min_alpha, 
            self.config.max_alpha, 
            self.config.n_alphas
        )
        
        best_alpha = 0.5
        best_score = -1.0
        
        self.train_results = {'alphas': [], 'scores': []}
        
        for alpha in alphas:
            score = self.evaluate_alpha(X_train, y_train, alpha)
            self.train_results['alphas'].append(alpha)
            self.train_results['scores'].append(score)
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
                
        self.best_alpha = best_alpha
        self.best_score = best_score
        
        return best_alpha
    
    def plot_learning_curve(self, save_path: Optional[str] = None) -> None:
        """Plot the learning curve showing performance vs. alpha.
        
        Args:
            save_path: Path to save the plot. If None, display the plot.
        """
        if not hasattr(self, 'train_results') or not self.train_results['alphas']:
            raise ValueError("No training results available. Call find_optimal_alpha first.")
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_results['alphas'], self.train_results['scores'], 'b-', label='Average Precision')
        plt.axvline(x=self.best_alpha, color='r', linestyle='--', 
                   label=f'Optimal α = {self.best_alpha:.3f}\nAP = {self.best_score:.3f}')
        
        plt.xlabel('Alpha (weight for semantic score)')
        plt.ylabel('Average Precision')
        plt.title('Optimal Alpha Selection')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved learning curve to {save_path}")
        else:
            plt.show()
    
    def save_results(self, output_dir: Optional[str] = None) -> None:
        """Save the learned weights and configuration.
        
        Args:
            output_dir: Directory to save results. If None, use config.output_dir
        """
        if self.best_alpha is None:
            raise ValueError("No weights have been learned. Call find_optimal_alpha first.")
            
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save weights
        weights = {
            'symbolic_weight': float(1 - self.best_alpha),
            'semantic_weight': float(self.best_alpha),
            'average_precision': float(self.best_score)
        }
        
        weights_path = os.path.join(output_dir, 'weights.json')
        with open(weights_path, 'w') as f:
            json.dump(weights, f, indent=2)
        
        logger.info(f"Saved learned weights to {weights_path}")
        
        # Save training curve
        plot_path = os.path.join(output_dir, 'learning_curve.png')
        self.plot_learning_curve(plot_path)
        
        # Save training data for reference
        train_data_path = os.path.join(output_dir, 'training_data.csv')
        if self.data is not None:
            self.data.to_csv(train_data_path, index=False)
            logger.info(f"Saved training data to {train_data_path}")
    
    def train(self, data_path: Optional[str] = None) -> Dict[str, float]:
        """Train the model to find optimal weights.
        
        Args:
            data_path: Path to the training data CSV file
            
        Returns:
            Dictionary containing the learned weights
        """
        # Load and prepare data
        self.load_data(data_path)
        X, y = self.prepare_features_labels()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Find optimal alpha
        best_alpha = self.find_optimal_alpha(X_train, y_train)
        test_score = self.evaluate_alpha(X_test, y_test, best_alpha)
        
        logger.info(f"Optimal alpha: {best_alpha:.4f}")
        logger.info(f"Training AP: {self.best_score:.4f}")
        logger.info(f"Test AP: {test_score:.4f}")
        
        # Save results
        self.save_results()
        
        return {
            'symbolic_weight': 1 - best_alpha,
            'semantic_weight': best_alpha,
            'train_ap': float(self.best_score),
            'test_ap': float(test_score)
        }

def main():
    """Main function to run weight learning."""
    # Initialize configuration
    config = TrainingConfig()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize and train the model
    learner = WeightLearner(config)
    weights = learner.train()
    
    print("\nTraining complete!")
    print(f"Optimal weights: symbolic={weights['symbolic_weight']:.3f}, semantic={weights['semantic_weight']:.3f}")
    print(f"Average Precision - Train: {weights['train_ap']:.3f}, Test: {weights['test_ap']:.3f}")
    print(f"Results saved to: {os.path.abspath(config.output_dir)}")

if __name__ == "__main__":
    main()
