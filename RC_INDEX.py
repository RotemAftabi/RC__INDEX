from RangeTree import RangeTree
from CoverTree import CoverTree
import numpy as np
import pandas as pd
import time
from scipy.spatial import distance

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]
data = pd.read_csv(url, header=None, names=columns, skipinitialspace=True)
df = pd.DataFrame(data)

# Select only numeric columns
numeric_columns = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
numeric_data = data[numeric_columns].values
randomly_chosen_columns = np.random.choice(numeric_columns, size=1, replace=False)
Adist = df[randomly_chosen_columns]
distances = distance.pdist(Adist, metric='euclidean')


class RCIndex:
    def __init__(self, data, b=2.0):
        """
        RC-Index that integrates Range Tree and Cover Trees.
        Args:
            data (np.array): The dataset on which the index is built.
            b (float): Base distance parameter for the Cover Trees.
        """
        start_time = time.time()
        print("Initializing RC-Index...")

        self.data = data  # שמירת הנתונים
        print(f"Dataset contains {len(data)} rows")

        # בניית ה-RangeTree
        print("Building RangeTree...")
        self.range_tree = RangeTree(data, b)
        print(f"RangeTree built in {time.time() - start_time:.2f} seconds")

        # בודקים אם הבעיה היא ב-CoverTree
        start_cover_time = time.time()
        print("Building CoverTree (this may take time)...")
        self.cover_tree = CoverTree(data, b) if len(data) > 1 else None
        print(f"Total RC-Index build time: {time.time() - start_time:.2f} seconds")

    def query(self, query_column, query_range, k, delta):
        """
        Executes a range query and returns k diverse results using Cover Trees.
        Args:
            query_column (str): The column to apply the range filter on.
            query_range (tuple): (min_val, max_val) range filter.
            k (int): Number of diverse results to return.
            delta (int): Depth parameter for extracting candidates.
        Returns:
            np.array: k diverse results.
        """
        column_index = numeric_columns.index(query_column)  # Find index of the column
        column_data = self.data[:, column_index]

        # Filter data based on the selected column range
        filtered_indices = np.where((column_data >= query_range[0]) & (column_data <= query_range[1]))[0]
        filtered_data = numeric_data[filtered_indices]

        # Build a temporary RCIndex for the filtered data
        relevant_cover_trees = self.range_tree.range_query(self.range_tree.root, query_range)
        candidates = []

        # Extract diverse candidates from each relevant Cover Tree
        for cover_tree in relevant_cover_trees:
            if cover_tree:
                candidates.extend(cover_tree.extract_candidates(k, delta))

        # Apply Greedy Tree++ selection algorithm
        return self.greedy_selection(candidates, k)

    def greedy_selection(self, candidates, k):
        """
        Selects k diverse items using Greedy Tree++.
        Args:
            candidates (list): List of candidate items.
            k (int): Number of items to select.
        Returns:
            np.array: k diverse results.
        """
        if not candidates:
            return np.array([])

        if len(candidates) < k:
            print(f"Warning: Only {len(candidates)} candidates found, but requested {k}. Returning all available.")
            return np.array(candidates)

        diverse_set = []
        candidates = list(candidates)  # Ensure it's mutable

        while len(diverse_set) < k and candidates:
            best_candidate = max(candidates, key=lambda x: min(
                np.linalg.norm(np.array(x) - np.array(y)) for y in diverse_set) if diverse_set else float('inf'))
            diverse_set.append(best_candidate)
            candidates.remove(best_candidate)

        return np.array(diverse_set)


# Example usage:
if __name__ == "__main__":
    # Initialize RC-Index with numeric data
    sample_data = numeric_data[:1000]
    rc_index = RCIndex(sample_data)

    # Get user input for query
    print("Available numeric columns:", numeric_columns)
    query_column = input("Enter the column name to apply range filter: ")
    if query_column not in numeric_columns:
        print("Invalid column. Exiting.")
        exit()

    min_val = float(input(f"Enter minimum value for {query_column}: "))
    max_val = float(input(f"Enter maximum value for {query_column}: "))
    query_range = (min_val, max_val)

    k = int(input("Enter number of diverse results to return (k): "))
    delta = int(input("Enter depth parameter for extracting candidates (delta): "))

    results = rc_index.query(query_column, query_range, k, delta)
    print("Diverse results:", results)
