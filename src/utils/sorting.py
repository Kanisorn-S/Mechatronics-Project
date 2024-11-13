import numpy as np

# Define collection zones for each size category
collection_zones = {"M3": (350, 0), "M4": (450, 0), "M5": (550, 0)}
free_level_y = 50  # Free level to avoid collection zones

# Function to map numeric size to string labels and sort nuts by x and y coordinates
def sort_nuts(xy_coords, nut_sizes):
    """
    Sort nuts by least x-coordinate, then by ascending y-coordinate within each x position.
    """
    # Map size indices to labels
    size_mapping = {0: "M3", 1: "M4", 2: "M5"}
    # Combine coordinates and mapped sizes into a list of tuples
    nuts = [(xy[0], xy[1], size_mapping[size]) for xy, size in zip(xy_coords, nut_sizes)]
    # Sort nuts by x-coordinate first, then by y-coordinate for an efficient path
    sorted_nuts = sorted(nuts, key=lambda nut: (nut[0], nut[1]))
    return sorted_nuts

# Function to generate the path for sweeping nuts to collection zones
def generate_sweep_path(nuts, collection_zones, free_level_y=50):
    """
    Generates a path for sweeping nuts of different sizes to designated collection zones.
    """
    path = [(0, 0)]  # Start at origin

    for nut in nuts:
        nut_x, nut_y, size = nut
        collection_x, collection_y = collection_zones[size]

        # Move up to a safe level above the nut
        path.append((0, nut_y + 10))          # Vertical move to safe level above nut
        path.append((nut_x, nut_y + 10))      # Horizontal move over the nut

        # Sweep down to free level
        path.append((nut_x, free_level_y))

        # Move horizontally to the collection zone's x position at free level
        path.append((collection_x, free_level_y))

        # Sweep nut down to the collection zone
        path.append((collection_x, collection_y))

        # Return to free level and then to (0, free_level_y) for the next sweep
        path.append((collection_x, free_level_y))
        path.append((0, free_level_y))

    # Return to starting position after all sweeps are complete
    path.append((0, 0))

    return path