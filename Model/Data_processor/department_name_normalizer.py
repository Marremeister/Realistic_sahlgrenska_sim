# Model/Data_processor/department_name_normalizer.py
"""
Shared utility class for normalizing department names consistently across the system.
"""
import os
import json
import logging
from collections import defaultdict


class DepartmentNameNormalizer:
    """
    Handles the normalization of department names to ensure consistency
    throughout the system. Used by multiple components including
    HospitalGraphBuilder and TimeBasedTransportAnalyzer.
    """

    def __init__(self, analysis_output_dir='analysis_output'):
        """
        Initialize the department name normalizer.

        Args:
            analysis_output_dir: Directory where normalization mapping files are stored
        """
        self.analysis_output_dir = analysis_output_dir
        self.name_mapping = {}
        self.normalized_departments = set()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Set up a logger for the DepartmentNameNormalizer."""
        logger = logging.getLogger("DepartmentNameNormalizer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def load_existing_mapping(self):
        """
        Try to load existing normalization mapping from files.

        Returns:
            bool: True if mapping was loaded, False otherwise
        """
        # Check for explicit name mapping file
        mapping_file = os.path.join(self.analysis_output_dir, 'name_mapping.json')
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    self.name_mapping = json.load(f)

                # Extract normalized departments
                self.normalized_departments = set(self.name_mapping.values())

                self.logger.info(
                    f"Loaded existing name mapping from {mapping_file} with {len(self.name_mapping)} entries")
                return True
            except Exception as e:
                self.logger.warning(f"Error loading name mapping from {mapping_file}: {str(e)}")

        # Check for node coordinates file as fallback
        coord_file = os.path.join(self.analysis_output_dir, 'node_coordinates.json')
        if os.path.exists(coord_file):
            try:
                with open(coord_file, 'r') as f:
                    coordinates = json.load(f)

                # Node names in coordinates are normalized; store them
                self.normalized_departments = set(coordinates.keys())
                self.logger.info(
                    f"Loaded {len(self.normalized_departments)} normalized department names from coordinates")
                return bool(self.normalized_departments)
            except Exception as e:
                self.logger.warning(f"Error loading node coordinates from {coord_file}: {str(e)}")

        self.logger.info("No existing name mapping found")
        return False

    def normalize_departments(self, departments):
        """
        Normalize a list of department names.

        Args:
            departments: List or set of department names to normalize

        Returns:
            dict: Mapping from original names to normalized names
        """
        # If we already have a mapping, use it to normalize
        if self.name_mapping:
            result = {}
            for dept in departments:
                if not dept or str(dept).strip() == "":
                    continue

                cleaned_name = str(dept).strip()
                if cleaned_name in self.name_mapping:
                    result[cleaned_name] = self.name_mapping[cleaned_name]
                else:
                    # For new departments, find best match from normalized departments
                    best_match = self._find_best_department_match(cleaned_name)
                    if best_match:
                        result[cleaned_name] = best_match
                    else:
                        result[cleaned_name] = cleaned_name
                        self.normalized_departments.add(cleaned_name)

            # Update our mapping with any new entries
            self.name_mapping.update(result)
            return result

        # If we have normalized departments but no mapping, use them for matching
        if self.normalized_departments:
            result = {}
            for dept in departments:
                if not dept or str(dept).strip() == "":
                    continue

                cleaned_name = str(dept).strip()
                best_match = self._find_best_department_match(cleaned_name)
                if best_match:
                    result[cleaned_name] = best_match
                else:
                    result[cleaned_name] = cleaned_name
                    self.normalized_departments.add(cleaned_name)

            # Store the new mapping
            self.name_mapping.update(result)
            return result

        # If we have nothing to go on, create mapping from scratch
        return self._create_new_mapping(departments)

    def _create_new_mapping(self, departments):
        """
        Create a completely new normalization mapping.

        Args:
            departments: List or set of department names to normalize

        Returns:
            dict: Mapping from original names to normalized names
        """
        name_mapping = {}
        normalized_names = set()
        department_groups = defaultdict(list)

        # First pass: group similar departments
        for dept in departments:
            if not dept or str(dept).strip() == "":
                continue

            cleaned_name = str(dept).strip()

            # Find the best match among existing groups
            best_match = None
            best_score = 0

            for norm_name in normalized_names:
                score = self._calculate_name_similarity(cleaned_name, norm_name)
                if score > 0.7 and score > best_score:  # Threshold of 0.7
                    best_score = score
                    best_match = norm_name

            if best_match:
                department_groups[best_match].append(cleaned_name)
                name_mapping[cleaned_name] = best_match
            else:
                normalized_names.add(cleaned_name)
                department_groups[cleaned_name].append(cleaned_name)
                name_mapping[cleaned_name] = cleaned_name

        # Store the results
        self.name_mapping = name_mapping
        self.normalized_departments = normalized_names

        # Log groups with multiple variations
        for group_name, members in department_groups.items():
            if len(members) > 1:
                self.logger.info(f"Normalized group '{group_name}' contains {len(members)} variations:")
                for member in members:
                    self.logger.info(f"  - {member}")

        self.logger.info(
            f"Created new name mapping with {len(departments)} names normalized to {len(normalized_names)} unique departments")
        return name_mapping

    def _find_best_department_match(self, dept_name, threshold=0.7):
        """
        Find the best match for a department name among normalized departments.

        Args:
            dept_name: Department name to match
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            str or None: Best matching department name or None if no good match
        """
        if not self.normalized_departments:
            return None

        # Try exact prefix match first (more reliable for hospital departments)
        # Get the first few words (e.g., "Öron Näsa Hals")
        words = dept_name.split()
        if len(words) >= 2:
            prefix = " ".join(words[:min(3, len(words))])

            for group in self.normalized_departments:
                group_words = group.split()
                if len(group_words) >= 2:
                    group_prefix = " ".join(group_words[:min(3, len(group_words))])

                    # If prefixes match, consider it the same department
                    if prefix.lower() == group_prefix.lower():
                        return group

        # If no prefix match, try more advanced similarity metrics
        best_match = None
        best_score = 0

        for group in self.normalized_departments:
            score = self._calculate_name_similarity(dept_name, group)

            if score > threshold and score > best_score:
                best_score = score
                best_match = group

        return best_match

    def _calculate_name_similarity(self, name1, name2):
        """
        Calculate similarity between two department names.

        Args:
            name1: First department name
            name2: Second department name

        Returns:
            float: Similarity score (0.0-1.0)
        """
        # Convert to lowercase for comparison
        name1 = str(name1).lower()
        name2 = str(name2).lower()

        # Simple character-based similarity
        # Compute Jaccard similarity of character sets
        set1 = set(name1)
        set2 = set(name2)

        if not set1 or not set2:
            return 0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        char_similarity = intersection / union

        # Word-based similarity
        words1 = set(name1.split())
        words2 = set(name2.split())

        if not words1 or not words2:
            return char_similarity  # Fall back to char similarity

        word_intersection = len(words1.intersection(words2))
        word_union = len(words1.union(words2))

        word_similarity = word_intersection / word_union

        # Combined score (weighted more toward word similarity)
        return 0.3 * char_similarity + 0.7 * word_similarity

    def get_normalized_name(self, original_name):
        """
        Get the normalized name for a given original name.

        Args:
            original_name: Original department name

        Returns:
            str: Normalized department name
        """
        if not original_name:
            return ""

        cleaned_name = str(original_name).strip()

        # If in mapping, return directly
        if cleaned_name in self.name_mapping:
            return self.name_mapping[cleaned_name]

        # Otherwise, normalize on the fly
        departments = [cleaned_name]
        self.normalize_departments(departments)
        return self.name_mapping.get(cleaned_name, cleaned_name)

    def save_mapping(self, custom_filename=None):
        """
        Save the current name mapping to a file.

        Args:
            custom_filename: Optional custom filename (default: name_mapping.json)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.name_mapping:
            self.logger.warning("No name mapping to save")
            return False

        try:
            if not os.path.exists(self.analysis_output_dir):
                os.makedirs(self.analysis_output_dir)

            filename = custom_filename or 'name_mapping.json'
            filepath = os.path.join(self.analysis_output_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(self.name_mapping, f, indent=2)

            self.logger.info(f"Saved name mapping to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving name mapping: {str(e)}")
            return False

    def get_mapping(self):
        """Get the current name mapping dictionary."""
        return dict(self.name_mapping)

    def get_normalized_departments(self):
        """Get the set of normalized department names."""
        return set(self.normalized_departments)