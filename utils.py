import functools
import time
import numpy as np
from typing import Optional
from termcolor import colored


def timeit(bold=False, dark=False):
    """`bold` and `dark` are used for printing"""
    def decorator(func):
        @functools.wraps(func)
        def timed(*args, **kw):
            ts = time.time()
            result = func(*args, **kw)
            te = time.time()

            formatted_time = f'{(te-ts) * 1000:,.2f} ms'
            if bold and dark:
                print_attrs = ("bold", "dark")
            elif bold:
                print_attrs = ("bold",)
            elif dark:
                print_attrs = ("dark",)
            else:
                print_attrs = ()

            print(colored(
                f"{func.__name__:<20} {formatted_time:>10}",
                attrs=print_attrs))

            return result
        return timed
    return decorator


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = '',
    suffix: str = '',
    decimals: int = 1,
    length: int = 100,
    fill: str = '█',
    print_end: str = "\r"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\\r", "\\r\\n") (Str)
    """
    percent = f"{100 * (iteration / float(total)):.2f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)

    # new line when done
    if iteration == total:
        print()


def convert_row_to_rank(
    row: np.ndarray, i: int, n_rows: int
) -> np.ndarray:
    """Return a ranked row where -1s stay as -1s and do not impact rank"""

    def sorted_position(arr, val, total_len):
        """Given that `arr` is sorted, return arr.index(val)"""
        lower = 0
        upper = total_len
        while lower <= upper:
            mid = (lower + upper) // 2
            if arr[mid] < val:
                lower = mid + 1
            elif arr[mid] > val:
                upper = mid - 1
            else:
                return mid
        return -1 * total_len  # so when we divide, it will still be -1

    def rank_value(val, sorted_vals, total_len):
        """Return a rank for a number given it's whole sorted row"""
        if val == -1:
            return -1
        return sorted_position(sorted_vals, val, total_len) / total_len

    print_progress_bar(
        i, n_rows - 1,
        prefix='Ranking rows:', suffix='Complete', length=50
    )

    sorted_vals = sorted(row)
    while sorted_vals[0] == -1:
        sorted_vals = sorted_vals[1:]
    total_len = len(sorted_vals) - 1
    return [rank_value(val, sorted_vals, total_len) for val in row]


@timeit(dark=True)
def convert_to_rank(M: np.ndarray) -> np.ndarray:
    """
    Return a rank matrix given a matrix of scores.
    If the score is -1, do not include it in the ranking (leave as -1).
    If the score is higher, it is ranked lower.
    """
    return np.array([
        convert_row_to_rank(row, i, M.shape[0]) for i, row in enumerate(M)
    ])


def matrix_mult(A, B):
    """Compute B @ A.T row by row"""
    for i, row in enumerate(B):
        yield [sum(row * a_row) for a_row in A]


if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8, 9], [21, 31, 41]])
    np.testing.assert_array_almost_equal(
        np.array(list(matrix_mult(a, b))),
        b @ a.T)
    np.testing.assert_array_almost_equal(
        np.array(list(matrix_mult(b, a))),
        a @ b.T)

    M = np.array([
        [-1, 2, 5, 1, 4, 4.5, -1],
        [-1, -1, 5, 1, 4, -1, -1],
        [-1, 2, -1, -1, 4, -1, -1]])
    rankM = np.array([
        [-1, 0.25, 1, 0, 0.5, 0.75, -1],
        [-1, -1, 1, 0, 0.5, -1, -1],
        [-1, 0, -1, -1, 1, -1, -1]
    ])
    np.testing.assert_array_almost_equal(convert_to_rank(M), rankM)
