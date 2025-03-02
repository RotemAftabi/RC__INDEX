from RangeTree import RangeTree
from CoverTree import CoverTree
import numpy as np
import pandas as pd
import time
import random
from scipy.spatial import distance
import sys

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]
chunk_size = 50000
data = pd.read_csv(url, header=None, names=columns, skipinitialspace=True, nrows=chunk_size)
df = pd.DataFrame(data)

# Select only numeric columns
numeric_columns = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
numeric_data = data[numeric_columns].values
randomly_chosen_columns = np.random.choice(numeric_columns, size=2, replace=False)
adist = [numeric_columns.index(randomly_chosen_columns[i]) for i in range(len(randomly_chosen_columns))]
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

        # Remove duplicates from data before constructing the index
        self.data = np.unique(data, axis=0)  # Removing duplicates from data
        if len(self.data) != len(data):
            print("Warning: Found duplicate rows in the dataset!")

        # בניית ה-RangeTree
        self.range_tree = RangeTree(data, b, adist, len(numeric_columns))
        print(f"RangeTree built in {time.time() - start_time:.2f} seconds")

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
        relevant_cover_trees = self.range_tree.range_query(self.range_tree.root, query_range, column_index)
        candidates = []

        # Extract diverse candidates from each relevant Cover Tree
        for cover_tree in relevant_cover_trees:
            if cover_tree:
                candidates.extend(cover_tree.extract_candidates(k, delta))
        candidates = set(tuple(i) for i in candidates)
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

        candidates = list(candidates)
        rand_inx = random.randint(0, len(candidates) - 1)
        random_candidate = candidates[rand_inx]
        candidates.pop(rand_inx)
        diverse_set = [random_candidate]
        distances = [np.linalg.norm(np.array(random_candidate) - np.array(item)) for item in candidates]
        while len(candidates) > 0 and len(diverse_set) < k:
            max_int = distances.index(max(distances))
            best_candidate = candidates[max_int]
            diverse_set.append(best_candidate)
            distances = [min(distances[i], np.linalg.norm(np.array(best_candidate) - np.array(candidates[i]))) for i in
                         range(len(candidates))]
        return np.array(diverse_set)

# Example usage:
if __name__ == "__main__":
    # Initialize RC-Index with numeric data
    sample_data = numeric_data[:chunk_size]
    rc_index = RCIndex(sample_data)

    # Get user input for query
    while True:
        print("Index chosen to build the tree:", adist)
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
        cont = input("would you like to continue? (y/n)\n")
        if cont == "n":
            break
