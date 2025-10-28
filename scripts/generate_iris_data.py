"""
Script to generate initial iris.csv file from scikit-learn.
Run this once to create the dataset.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import get_iris_data


def main():
    """Generate iris.csv file."""
    print("Generating iris.csv from scikit-learn...")
    
    try:
        # This will load from sklearn and save to CSV
        X, y = get_iris_data(use_csv=False)
        print(f" Successfully generated iris.csv with {len(X)} samples")
        print(f" Features: {list(X.columns)}")
        print(f" Classes: {y.nunique()}")
        
    except Exception as e:
        print(f" Error generating dataset: {str(e)}")


if __name__ == "__main__":
    main()