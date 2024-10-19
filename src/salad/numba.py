import numba

def _xyz_to_xy_prime(x: np.ndarray, b: np.ndarray, reference_time: np.float64):
    vx = b[0]
    vy = b[1]
    x_prime = x[0] - vx * (x[2] - reference_time)
    y_prime = x[1] - vy * (x[2] - reference_time)
    return x_prime, y_prime

_numba_xyz_to_xy_prime = numba.njit(_xyz_to_xy_prime)

def _digitize_point(p, min_x, min_y, dx, dy):
    return int((p[0] - min_x)/dx), int((p[1] - min_y)/dy)

_numba_digitize_point = numba.njit(_digitize_point)

def _vote_points(hough: np.ndarray, X: np.ndarray, b: np.ndarray, x_min, y_min, dx, dy, reference_time, coef: np.float64=1):
    """
    Given detections vote in the hough space
    """
    n, d = X.shape
    num_b = b.shape[0]
    for b_idx in numba.prange(num_b): # for each direction
        if b_idx < num_b:
            b_i = b[b_idx]
            for i in range(n): # for each point
                x_prime, y_prime = xyz_to_xy_prime(X[i], b_i, reference_time)
                x_idx, y_idx = digitize_point(np.array([x_prime, y_prime]), x_min, y_min, dx, dy)
                if x_idx < hough.shape[1] and y_idx < hough.shape[2]:
                    hough[b_idx, x_idx, y_idx] += coef

_numba_vote_points = numba.njit(_vote_points, parallel=True)   
                    
def _vote_bins(hough: np.ndarray, bins: np.ndarray, coef: np.float64=1):
    """
    Given pre-binned detections vote in the hough space
    """
    num_b, n, _ = bins.shape
    for b_idx in numba.prange(num_b): # for each direction
        for i in range(n): # for each point
            if b_idx < num_b:
                x_idx = bins[b_idx, i, 0] # bin location for that point
                y_idx = bins[b_idx, i, 1]
                if x_idx < hough.shape[1] and y_idx < hough.shape[2]:
                    hough[b_idx, x_idx, y_idx] += coef

_numba_vote_bins = numba.njit(_vote_bins, parallel=True)


def _transform_to_xy_prime(X: np.ndarray, b: np.ndarray, reference_time: np.float64):
    num_b = b.shape[0]
    n = X.shape[0]
    M = np.zeros((num_b, n, 2))
    for i in numba.prange(num_b):
        for j in range(n):
            M[i, j] = xyz_to_xy_prime(X[j], b[i], reference_time)
    return M

_numba_transform_to_xy_prime = numba.njit(_transform_to_xy_prime, parallel=True)

def _digitize_xy(points, min_x, min_y, dx, dy):
    bins = np.zeros(points.shape, dtype=np.int32)
    for i in numba.prange(points.shape[0]):
        for j in range(points.shape[1]):
            x, y = digitize_point(points[i, j], min_x, min_y, dx, dy)
            bins[i, j, 0] = x
            bins[i, j, 1] = y
    return bins

_numba_digitize_xy = numba.njit(_digitize_xy, parallel=True)


def _close_to_line(X, anchor, direction, dx, reference_time):
    M = _numba_transform_to_xy_prime(X, np.atleast_2d(direction), reference_time)[0]
    d = (anchor - M)
    d_norm = np.sum(d**2, axis=1)**0.5
    return d_norm <= dx

_numba_close_to_line = numba.njit(_close_to_line)

# directions.b.value