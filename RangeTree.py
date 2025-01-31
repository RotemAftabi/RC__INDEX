import data

from CoverTree import CoverTree
import numpy as np
from scipy.spatial import distance
import time


class RangeTreeNode:
    def __init__(self, data, b=2.0):
        self.data = np.array(data)
        self.cover_tree = CoverTree(data, b) if len(data) > 1 else None
        self.left = None
        self.right = None
        self.median = None


class RangeTree:
    def __init__(self, data, b=2.0):
        start_time = time.time()
        print(f"Building RangeTree for {len(data)} rows...")
        self.root = self._build_tree(data, b)
        print(f"RangeTree built in {time.time() - start_time:.2f} seconds")

    def _build_tree(self, data, b):
        start_time = time.time()
        print(f"Building tree with {len(data)} rows...")

        if len(data) == 0:
            return None
        if len(data) == 1:
            return RangeTreeNode(data, b)
        if len(data) > 5000:
            print(f"Warning: Large node detected ({len(data)} rows). Consider optimization!")

        sorted_indices = np.argsort(data[:, 0])
        sorted_data = data[sorted_indices]
        median_index = len(sorted_data) // 2
        median = sorted_data[median_index]

        print(f"Splitting at median value: {median[0]} (size={len(sorted_data)})")

        node = RangeTreeNode(sorted_data, b)
        node.median = median

        left_start = time.time()
        node.left = self._build_tree(sorted_data[:median_index], b)
        print(f"Left subtree built in {time.time() - left_start:.4f} seconds")

        right_start = time.time()
        node.right = self._build_tree(sorted_data[median_index:], b)
        print(f"Right subtree built in {time.time() - right_start:.4f} seconds")

        print(f"Tree node built in {time.time() - start_time:.4f} seconds")
        return node

    def range_query(self, node, query_range):
        if node is None:
            return []
        min_val, max_val = query_range
        median_value = node.median[0]
        if max_val < median_value:
            return self.range_query(node.left, query_range)
        elif min_val > median_value:
            return self.range_query(node.right, query_range)
        else:
            results = [node.cover_tree] if node.cover_tree else []
            results += self.range_query(node.left, query_range)
            results += self.range_query(node.right, query_range)
            return results

    def query_diverse_results(self, query_range, k, delta):
        relevant_trees = self.range_query(self.root, query_range)
        candidates = []
        for tree in relevant_trees:
            if tree:
                candidates.extend(tree.extract_candidates(k, delta))
        return self.greedy_selection(candidates, k)

    def greedy_selection(self, candidates, k):
        if not candidates:
            print("No candidates found for greedy selection.")
            return []

        diverse_set = []
        candidates = list(candidates)

        while len(diverse_set) < k and candidates:
            best_candidate = max(candidates, key=lambda x: min(
                distance.euclidean(x, y) for y in diverse_set) if diverse_set else float('inf'))
            diverse_set.append(best_candidate)
            candidates.remove(best_candidate)

        return diverse_set
