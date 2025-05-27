import numpy as np
from scipy.spatial import Delaunay # Used for point_in_triangle check
from shapely.geometry import Point, Polygon # Used for distance to triangle calculation

def get_triangle_area(vertices: np.ndarray) -> float:
    """
    Calculates the area of a triangle given its vertices.
    Args:
        vertices: A NumPy array of shape (3, 2) representing the triangle's vertices.
                  Example: np.array([[x1, y1], [x2, y2], [x3, y3]])
    Returns:
        The area of the triangle (float).
    """
    if not isinstance(vertices, np.ndarray) or vertices.shape != (3, 2):
        raise TypeError("Vertices must be a NumPy array of shape (3, 2).")
    # Using the shoelace formula
    area = 0.5 * abs(
        vertices[0, 0] * (vertices[1, 1] - vertices[2, 1]) +
        vertices[1, 0] * (vertices[2, 1] - vertices[0, 1]) +
        vertices[2, 0] * (vertices[0, 1] - vertices[1, 1])
    )
    if area == 0:
        raise ValueError("The vertices do not form a valid triangle (area is zero).")
    return area

def distance_point_to_triangle(point: np.ndarray, triangle_poly: Polygon) -> float:
    """
    Calculates the shortest Euclidean distance from a point to a triangle.
    If the point is inside or on the boundary of the triangle, the distance is 0.

    Args:
        point: A 1D NumPy array representing the point (e.g., np.array([x, y])).
        triangle_poly: A Shapely Polygon object representing the triangle.

    Returns:
        The shortest distance from the point to the triangle (float).
    """
    p = Point(point)
    if triangle_poly.contains(p) or triangle_poly.touches(p):
        return 0.0
    return p.distance(triangle_poly)


def calculate_tah_score(points: np.ndarray, triangle_vertices: np.ndarray) -> float:
    """
    Calculates the Triangle Attractiveness and Homogeneity (TAH) score.

    The TAH score ranges from 0 to 1, where 1 represents an ideal
    distribution (all points homogeneously spread within the triangle).

    Args:
        points: A 2D NumPy array of shape (n, 2) representing the n points,
                where each row is [x, y].
        triangle_vertices: A 2D NumPy array of shape (3, 2) representing
                           the vertices of the triangle [[x1,y1], [x2,y2], [x3,y3]].

    Returns:
        The TAH score (float between 0 and 1).

    Raises:
        TypeError: If inputs are not of the expected types or convertible.
        ValueError: If triangle_vertices do not form a valid triangle,
                    or if points array is not 2D with 2 columns.
    """

    # --- Input Validation ---
    if not isinstance(points, np.ndarray):
        raise TypeError("Input 'points' must be a NumPy array.")
    if not isinstance(triangle_vertices, np.ndarray):
        raise TypeError("Input 'triangle_vertices' must be a NumPy array.")

    if triangle_vertices.shape != (3, 2):
        raise ValueError("Triangle vertices must be a NumPy array of shape (3, 2).")

    if points.ndim != 2 or (points.shape[1] != 2 and points.size != 0) : # Allow empty array of shape (0,0) or (0,2)
         raise ValueError("Input 'points' must be a 2D array with 2 columns (x, y) or an empty array.")

    n = points.shape[0]

    # --- Triangle Properties ---
    try:
        triangle_area_AT = get_triangle_area(triangle_vertices)
    except ValueError as e:
        raise ValueError(f"Invalid triangle_vertices: {e}")

    triangle_polygon = Polygon(triangle_vertices)

    # --- I. Triangle Attractiveness Weight Function (w(p_i)) ---
    # Described in TAH.md
    if n == 0:
        point_weights = np.array([], dtype=float)
    else:
        # Calculate distance d_T(p_i) for each point p_i from the triangle T
        # d_T(p_i) = 0 if p_i is inside or on the boundary of T
        # d_T(p_i) = shortest Euclidean distance if p_i is outside T
        d_T_pi = np.array([distance_point_to_triangle(p, triangle_polygon) for p in points])

        # Calculate weights w(p_i) = exp( -2 * (d_T(p_i))^2 / A_T )
        # A_T is the 2D analog to L^2 in 1D SAH.py
        point_weights = np.exp(-2 * (d_T_pi ** 2) / triangle_area_AT)

    # --- II. Calculating the TAH Score ---

    # 1. Calculate Mean Attractiveness (MA)
    # If n = 0, MA = 0.0
    # Otherwise, MA = (1/n) * sum(w(p_i))
    if n == 0:
        mean_attractiveness_MA = 0.0
    else:
        mean_attractiveness_MA = np.sum(point_weights) / n

    # 2. Calculate Weighted Homogeneity (HW)
    # If n <= 1, HW = 1.0 (based on SAH.py logic)
    weighted_homogeneity_HW = 1.0

    if n > 1:
        W_sum = np.sum(point_weights)

        # If W_sum is close to 0, HW = 1.0 (based on SAH.py logic)
        if np.isclose(W_sum, 0):
            weighted_homogeneity_HW = 1.0
        else:
            # Calculate nearest neighbor distances (d_nn_i) for each point p_i
            nn_dists = np.zeros(n)
            if n > 1: # Need at least two points to calculate distances
                from sklearn.neighbors import NearestNeighbors
                # Find the 2 nearest neighbors (the point itself and its actual nearest neighbor)
                nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points)
                distances, indices = nbrs.kneighbors(points)
                nn_dists = distances[:, 1] # distance to the *actual* nearest neighbor (not itself)

            # Weighted mean of nearest neighbor distances (mu_d)
            # mu_d = sum(d_nn_i * w_i) / W_sum
            # Reference to sorted weights in SAH.py is noted but adapted for 2D by summing over all points.
            mu_d = np.sum(nn_dists * point_weights) / W_sum

            # Weighted variance of nearest neighbor distances (sigma_d_sq)
            # sigma_d_sq = sum(((d_nn_i - mu_d)^2) * w_i) / W_sum
            variance_terms = ((nn_dists - mu_d) ** 2) * point_weights
            sigma_d_sq = np.sum(variance_terms) / W_sum
            sigma_d_sq = np.maximum(0, sigma_d_sq) # Ensure non-negativity

            # Weighted coefficient of variation (CV_d)
            if np.isclose(mu_d, 0): # Implies all relevant points are co-located
                cv_d = 0.0
            else:
                cv_d = np.sqrt(sigma_d_sq) / mu_d

            # Homogeneity score HW
            weighted_homogeneity_HW = 1.0 / (1.0 + cv_d)

    # 3. Final TAH Score
    # TAH = MA * HW
    # The SAH score structure is preserved.
    # A TAH score of 1 represents an ideal distribution.
    tah_score = mean_attractiveness_MA * weighted_homogeneity_HW

    return tah_score


if __name__ == "__main__":

    # Example: Define points and triangle vertices
    points_example = np.array([
        [0.5, 0.5],  # Inside
        [0.2, 0.2],  # Inside
        [0.8, 0.1],  # Inside
        [1.5, 1.5],  # Outside
        [0.5, 0.0],  # On edge
        [-0.5, 0.5]  # Outside
    ])

    triangle_verts_example = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0]
    ])

    # Handle case with no points
    empty_points = np.array([]).reshape(0,2) # Correctly shaped empty array
    tah_score_empty = calculate_tah_score(empty_points, triangle_verts_example)
    print(f"TAH Score for empty points: {tah_score_empty}")

    # Handle case with one point
    one_point = np.array([[0.5, 0.5]])
    tah_score_one_point = calculate_tah_score(one_point, triangle_verts_example)
    print(f"TAH Score for one point: {tah_score_one_point}")


    # Calculate TAH score
    try:
        tah_score_example = calculate_tah_score(points_example, triangle_verts_example)
        print(f"Calculated TAH Score: {tah_score_example}")

        # Example with points far away, expecting low MA and thus low TAH
        points_far = np.array([
            [10.0, 10.0],
            [12.0, 11.0],
            [9.0, 10.5]
        ])
        tah_score_far = calculate_tah_score(points_far, triangle_verts_example)
        print(f"Calculated TAH Score (far points): {tah_score_far}")

        # Example with points inside but clustered, expecting high MA but lower HW
        points_clustered_inside = np.array([
            [0.5, 0.5],
            [0.51, 0.51],
            [0.49, 0.49],
            [0.5, 0.48]
        ])
        tah_score_clustered = calculate_tah_score(points_clustered_inside, triangle_verts_example)
        print(f"Calculated TAH Score (clustered inside): {tah_score_clustered}")

        # Example with points perfectly homogeneous inside (hypothetical perfect score)
        # For a perfect score, MA=1 (all points inside or very close with high weight)
        # and HW=1 (CVd=0, meaning all weighted nearest neighbor distances are effectively equal)
        # This is harder to achieve perfectly with few points but let's try a somewhat ideal setup
        points_ideal = np.array([
            [0.25, 0.25], # Spread out inside
            [0.75, 0.25],
            [0.5,  0.75]
        ])
        tah_score_ideal = calculate_tah_score(points_ideal, triangle_verts_example)
        print(f"Calculated TAH Score (more ideal inside): {tah_score_ideal}")


    except (TypeError, ValueError) as e:
        print(f"Error: {e}")