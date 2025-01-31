import numpy as np
from scipy.spatial import distance

class CoverTree:
    def __init__(self, data, b=2.0):
        """
        Initializes the Cover Tree with hierarchical levels based on distances.
        Args:
            data (np.array): Dataset to build the Cover Tree from.
            b (float): Base factor that determines the distance scale between levels.
        """
        self.data = np.array(data)
        self.b = b
        self.layers, self.parent_child_map = self._build_tree()

    def _build_tree(self):
        """
        Build hierarchical layers for the Cover Tree with proper nesting and parent-child relationships.
        Returns:
            dict: Nested levels from l=0 down to l_min.
        """
        if len(self.data) == 0:
            return {}, {}

        # Compute min distance to determine l_min
        min_dist = min(
            distance.euclidean(self.data[i], self.data[j])
            for i in range(len(self.data)) for j in range(i + 1, len(self.data))
        ) if len(self.data) > 1 else 1.0
        l_min = int(np.floor(np.log(min_dist) / np.log(self.b)))  # יכול להיות מספר שלילי

        # Initialize levels {l=0, l=-1, ..., l_min}
        levels = {l: [] for l in range(0, l_min - 1, -1)}  # יורד מ-l=0 ל-l_min
        parent_child_map = {}

        # בוחרים נקודת עוגן ראשונית (למשל, מרכז המסה)
        centroid = np.mean(self.data, axis=0)

        # סידור הנתונים לפי מרחק ממרכז המסה
        distances = np.array([distance.euclidean(row, centroid) for row in self.data])
        sorted_indices = np.argsort(distances)
        sorted_data = self.data[sorted_indices]

        # יצירת היררכיה מהרמה העליונה למטה
        current_layer = list(sorted_data)
        for l in range(0, l_min - 1, -1):  # יורדים ברמות
            threshold = self.b ** l
            new_layer = []
            level_parent_map = {}

            for row in current_layer:
                added = False
                for parent in new_layer:
                    if distance.euclidean(row, parent) <= threshold:
                        level_parent_map.setdefault(tuple(parent), []).append(tuple(row))
                        added = True
                        break

                if not added:
                    new_layer.append(row)
                    level_parent_map[tuple(row)] = []

            levels[l] = new_layer
            parent_child_map.update(level_parent_map)
            current_layer = new_layer  # ממשיכים לרמה הבאה

        return levels, parent_child_map

    def extract_candidates(self, k, delta):
        """
        Extract candidates based on the k-level rule: Find the first level with at least k points,
        then descend delta levels below it while ensuring no duplicates.
        Args:
            k (int): Minimum number of candidates required.
            delta (int): Number of levels to explore below the found k-level.
        Returns:
            np.array: Candidate set (without duplicates).
        """
        k_level = None
        for l in self.layers.keys():
            if len(self.layers[l]) >= k:
                k_level = l
                break

        if k_level is None:
            return np.array([])  # No valid level found

        # מציאת מועמדים מהשכבות שמתחת לרמת k-level
        seen = set()
        candidates = []

        for l in range(k_level, max(k_level - delta - 1, min(self.layers.keys()) - 1), -1):
            for point in self.layers[l]:
                point_tuple = tuple(point)
                if point_tuple not in seen:
                    seen.add(point_tuple)
                    candidates.append(point)

        return np.array(candidates)
