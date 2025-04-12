#!/usr/bin/env python3
"""
Debugging utility to export and compare benchmark requests with hospital graph.
This helps identify department name mismatches between the requests and graph.
"""

import os
import json
import sys


def export_benchmark_requests(requests, output_file='debug_requests.json'):
    """
    Export generated benchmark requests to a JSON file for debugging.

    Args:
        requests: List of (origin, destination, transport_type, urgent) tuples
        output_file: Path to output file

    Returns:
        str: Path to output file
    """
    # Convert requests to a more readable format
    request_data = [
        {
            "origin": origin,
            "destination": dest,
            "transport_type": t_type,
            "urgent": urgent
        }
        for origin, dest, t_type, urgent in requests
    ]

    # Extract unique department names
    departments = set()
    for req in request_data:
        departments.add(req["origin"])
        departments.add(req["destination"])

    # Create complete export data
    export_data = {
        "requests": request_data,
        "unique_departments": sorted(list(departments)),
        "count": len(request_data)
    }

    # Create directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported {len(request_data)} requests with {len(departments)} unique departments to {output_file}")
    return output_file


def compare_with_hospital_graph(requests_file='debug_requests.json', graph_file='analysis_output/hospital_graph.json'):
    """
    Compare department names in requests with those in the hospital graph.

    Args:
        requests_file: Path to exported requests file
        graph_file: Path to hospital graph file

    Returns:
        tuple: (matching_depts, missing_depts, extra_depts)
    """
    # Load files
    try:
        with open(requests_file, 'r') as f:
            requests_data = json.load(f)

        with open(graph_file, 'r') as f:
            graph_data = json.load(f)
    except Exception as e:
        print(f"Error loading files: {str(e)}")
        return [], [], []

    # Extract department sets
    request_depts = set(requests_data["unique_departments"])
    graph_depts = set(graph_data.get("departments", []))

    # Find differences
    matching_depts = request_depts.intersection(graph_depts)
    missing_depts = request_depts - graph_depts
    extra_depts = graph_depts - request_depts

    # Print report
    print("\nDepartment Name Comparison:")
    print(f"  - Matching departments: {len(matching_depts)}")
    print(f"  - Departments in requests but NOT in graph: {len(missing_depts)}")
    print(f"  - Departments in graph but not in requests: {len(extra_depts)}")

    if missing_depts:
        print("\nMISSING DEPARTMENTS (in requests but not in graph):")
        for dept in sorted(missing_depts):
            print(f"  - {dept}")

    return matching_depts, missing_depts, extra_depts


if __name__ == "__main__":
    if len(sys.argv) > 1:
        compare_with_hospital_graph(sys.argv[1])
    else:
        compare_with_hospital_graph()