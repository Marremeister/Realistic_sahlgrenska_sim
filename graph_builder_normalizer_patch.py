#!/usr/bin/env python3
"""
Patch script to update HospitalGraphBuilder to use the shared DepartmentNameNormalizer.
This script modifies the graph_builder.py file in-place to replace the department
normalization logic with calls to the shared normalizer.

Usage:
    python graph_builder_normalizer_patch.py
"""
import os
import re
import shutil
import sys


def patch_graph_builder():
    """
    Patch the HospitalGraphBuilder class to use DepartmentNameNormalizer.
    """
    # Path to the graph_builder.py file
    file_path = os.path.join('Model', 'Data_processor', 'graph_builder.py')

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run this script from the project root.")
        return False

    # Create backup
    backup_path = file_path + '.bak'
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")

    # Read file content
    with open(file_path, 'r') as f:
        content = f.read()

    # Add import for DepartmentNameNormalizer
    import_pattern = re.compile(r'from Model\.model_pathfinder import Pathfinder')
    import_replacement = 'from Model.model_pathfinder import Pathfinder\nfrom Model.Data_processor.department_name_normalizer import DepartmentNameNormalizer'

    if import_pattern.search(content):
        content = import_pattern.sub(import_replacement, content)
    else:
        # If pattern not found, add import at the top after other imports
        content = content.replace('import random',
                                  'import random\nfrom Model.Data_processor.department_name_normalizer import DepartmentNameNormalizer',
                                  1)

    # Replace _normalize_department_names method
    normalize_pattern = re.compile(
        r'def _normalize_department_names\(self, departments\):(.*?)def _add_departments_as_nodes', re.DOTALL)

    normalize_replacement = '''def _normalize_department_names(self, departments):
        """
        Normalize department names by grouping similar names together using shared normalizer.

        Args:
            departments: List of original department names

        Returns:
            tuple: (normalized_names, name_mapping)
                - normalized_names: List of unique normalized department names
                - name_mapping: Dictionary mapping original names to normalized names
        """
        self.logger.info("Normalizing department names...")

        # Use the shared normalizer
        normalizer = DepartmentNameNormalizer(self.analysis_output_dir if hasattr(self, 'analysis_output_dir') else 'analysis_output')

        # First try to load existing mapping
        normalizer.load_existing_mapping()

        # Then normalize our departments
        name_mapping = normalizer.normalize_departments(departments)

        # Extract normalized names
        normalized_names = list(set(name_mapping.values()))

        # Save the mapping for future use
        normalizer.save_mapping()

        self.logger.info(f"Reduced {len(departments)} department names to {len(normalized_names)} unique departments")

        return normalized_names, name_mapping

    def _add_departments_as_nodes'''

    if normalize_pattern.search(content):
        content = normalize_pattern.sub(normalize_replacement, content)

        # Remove the _find_best_department_match and _calculate_name_similarity methods
        find_best_match_pattern = re.compile(
            r'def _find_best_department_match\(self, dept_name, existing_groups, threshold=0\.7\):(.*?)def _calculate_name_similarity',
            re.DOTALL)
        calculate_similarity_pattern = re.compile(
            r'def _calculate_name_similarity\(self, name1, name2\):(.*?)def _add_departments_as_nodes', re.DOTALL)

        if find_best_match_pattern.search(content) and calculate_similarity_pattern.search(content):
            # Remove these methods - they're now handled by the normalizer
            content = find_best_match_pattern.sub('def _add_departments_as_nodes', content)
            content = calculate_similarity_pattern.sub('def _add_departments_as_nodes', content)
        else:
            print("Warning: Could not find methods to remove. Manual cleanup may be needed.")
    else:
        print("Warning: Could not find _normalize_department_names method to replace.")

    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"Successfully patched {file_path} to use DepartmentNameNormalizer")
    print("Note: This is a basic patch. Manual review and testing is recommended.")
    return True


if __name__ == "__main__":
    if patch_graph_builder():
        print("\nPatching complete! Please review and test the changes.")
    else:
        print("\nPatching failed. The original file was not modified.")