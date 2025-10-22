"""
Geometry utilities for Traffic Signal Violation Detection System.

This module provides geometric calculation functions, primarily for
determining line-segment intersections used in violation detection.
"""

from typing import Tuple

# Type alias for improved readability
Point = Tuple[int, int]


def line_segment_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    Determine if two line segments intersect.

    This function checks whether the line segment (p1, p2) intersects with
    the line segment (p3, p4). Used for detecting when vehicle bounding boxes
    cross the traffic signal line.

    Args:
        p1: First endpoint of first line segment (x, y)
        p2: Second endpoint of first line segment (x, y)
        p3: First endpoint of second line segment (x, y)
        p4: Second endpoint of second line segment (x, y)

    Returns:
        bool: True if the line segments intersect, False otherwise

    Algorithm:
        Uses the general form of line equations (ax + by = c) to find
        the intersection point, then checks if that point lies within
        both line segments.

    Example:
        >>> line_segment_intersection((0, 0), (10, 10), (0, 10), (10, 0))
        True  # Perpendicular lines crossing at (5, 5)

        >>> line_segment_intersection((0, 0), (5, 5), (10, 0), (15, 5))
        False  # Parallel lines, no intersection
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Convert to general form: ax + by = c
    # For line through (x1, y1) and (x2, y2):
    # a1 = y1 - y2, b1 = x2 - x1, c1 = x1*y2 - x2*y1
    a1 = y1 - y2
    b1 = x2 - x1
    c1 = x1 * y2 - x2 * y1

    # For line through (x3, y3) and (x4, y4):
    a2 = y3 - y4
    b2 = x4 - x3
    c2 = x3 * y4 - x4 * y3

    # Check if lines are parallel (determinant == 0)
    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        # Lines are parallel or coincident
        return False

    # Solve for intersection point using Cramer's rule
    x = (b1 * c2 - b2 * c1) / determinant
    y = (a2 * c1 - a1 * c2) / determinant

    # Ensure coordinates are ordered (min, max) for each line segment
    x1_min, x1_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y1_min, y1_max = (y1, y2) if y1 <= y2 else (y2, y1)
    x2_min, x2_max = (x3, x4) if x3 <= x4 else (x4, x3)
    y2_min, y2_max = (y3, y4) if y3 <= y4 else (y4, y3)

    # Check if intersection point lies within both line segments
    within_segment_1 = x1_min <= x <= x1_max and y1_min <= y <= y1_max
    within_segment_2 = x2_min <= x <= x2_max and y2_min <= y <= y2_max

    return within_segment_1 and within_segment_2


def bounding_box_intersects_line(
    bbox_min: Point,
    bbox_max: Point,
    line_p1: Point,
    line_p2: Point
) -> bool:
    """
    Check if a bounding box intersects with a line segment.

    This is the primary function used for traffic violation detection.
    It checks if any of the four edges of a bounding box intersect
    with the traffic signal line.

    Args:
        bbox_min: Top-left corner of bounding box (xmin, ymin)
        bbox_max: Bottom-right corner of bounding box (xmax, ymax)
        line_p1: First endpoint of line segment
        line_p2: Second endpoint of line segment

    Returns:
        bool: True if any edge of the bounding box crosses the line

    Example:
        >>> # Vehicle bbox crossing traffic line
        >>> bounding_box_intersects_line(
        ...     (100, 100), (200, 200),  # Vehicle bounding box
        ...     (0, 150), (300, 150)      # Horizontal traffic line
        ... )
        True
    """
    xmin, ymin = bbox_min
    xmax, ymax = bbox_max

    # Define the four edges of the bounding box
    top_edge = ((xmin, ymin), (xmax, ymin))
    bottom_edge = ((xmin, ymax), (xmax, ymax))
    left_edge = ((xmin, ymin), (xmin, ymax))
    right_edge = ((xmax, ymin), (xmax, ymax))

    # Check intersection with each edge
    edges = [top_edge, bottom_edge, left_edge, right_edge]

    for edge_p1, edge_p2 in edges:
        if line_segment_intersection(line_p1, line_p2, edge_p1, edge_p2):
            return True

    return False


def point_distance(p1: Point, p2: Point) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        float: Distance between the points

    Example:
        >>> point_distance((0, 0), (3, 4))
        5.0
    """
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def is_point_in_bounding_box(point: Point, bbox_min: Point, bbox_max: Point) -> bool:
    """
    Check if a point is inside a bounding box.

    Args:
        point: Point to check (x, y)
        bbox_min: Top-left corner of bounding box
        bbox_max: Bottom-right corner of bounding box

    Returns:
        bool: True if point is inside the bounding box

    Example:
        >>> is_point_in_bounding_box((50, 50), (0, 0), (100, 100))
        True
        >>> is_point_in_bounding_box((150, 50), (0, 0), (100, 100))
        False
    """
    px, py = point
    xmin, ymin = bbox_min
    xmax, ymax = bbox_max

    return xmin <= px <= xmax and ymin <= py <= ymax


if __name__ == "__main__":
    # Unit tests when run directly
    print("Running geometry module tests...")

    # Test 1: Perpendicular lines that intersect
    assert line_segment_intersection((0, 5), (10, 5), (5, 0), (5, 10)) == True
    print("✓ Test 1 passed: Perpendicular lines intersection")

    # Test 2: Parallel lines that don't intersect
    assert line_segment_intersection((0, 0), (10, 0), (0, 5), (10, 5)) == False
    print("✓ Test 2 passed: Parallel lines no intersection")

    # Test 3: Non-intersecting segments
    assert line_segment_intersection((0, 0), (5, 5), (10, 0), (15, 5)) == False
    print("✓ Test 3 passed: Non-intersecting segments")

    # Test 4: Bounding box crossing line
    assert bounding_box_intersects_line((100, 100), (200, 200), (0, 150), (300, 150)) == True
    print("✓ Test 4 passed: Bounding box crosses line")

    # Test 5: Bounding box not crossing line
    assert bounding_box_intersects_line((100, 100), (200, 200), (0, 50), (300, 50)) == False
    print("✓ Test 5 passed: Bounding box doesn't cross line")

    # Test 6: Point distance
    assert point_distance((0, 0), (3, 4)) == 5.0
    print("✓ Test 6 passed: Point distance calculation")

    # Test 7: Point in bounding box
    assert is_point_in_bounding_box((50, 50), (0, 0), (100, 100)) == True
    assert is_point_in_bounding_box((150, 50), (0, 0), (100, 100)) == False
    print("✓ Test 7 passed: Point in bounding box check")

    print("\n✅ All geometry tests passed!")
