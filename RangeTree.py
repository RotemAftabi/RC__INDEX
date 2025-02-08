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
        if len(data) == 0:
            return None

        if len(data) == 1:
            node = RangeTreeNode(data, b)
            node.median = data[0]
            return node

        if len(data) < 50:
            node = RangeTreeNode(data, b)
            node.median = data[len(data) // 2]
            return node

        sorted_indices = np.argsort(data[:, 0])
        sorted_data = data[sorted_indices]

        median_index = len(sorted_data) // 2
        median = sorted_data[median_index]

        node = RangeTreeNode(sorted_data, b)
        node.median = median

        node.left = self._build_tree(sorted_data[:median_index], b)
        node.right = self._build_tree(sorted_data[median_index:], b)
        return node

    def range_query(self, node, query_range):
        if node is None:
            return []

        if node.median is None:
            print("Warning: Encountered node with no median!")
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
