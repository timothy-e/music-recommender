import functools
import time
import numpy as np
from termcolor import colored


def timeit(bold=False, dark=False):
    """`bold` and `dark` are used for printing"""
    def decorator(func):
        @functools.wraps(func)
        def timed(*args, **kw):
            ts = time.time()
            result = func(*args, **kw)
            te = time.time()

            formatted_time = f'{(te-ts) * 100:,.2f} ms'
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


@timeit(dark=True)
def convert_to_rank(M: np.ndarray) -> np.ndarray:
    """
    Return a rank matrix given a matrix of scores.
    If the score is -1, do not include it in the ranking (leave as -1).
    If the score is higher, it is ranked lower.
    """
    def rank_value(val, sorted_vals):
        if val == -1:
            return -1
        return sorted_vals.index(val) / (len(sorted_vals) - 1)

    def rank_row(row):
        sorted_vals = sorted(row)
        while sorted_vals[0] == -1:
            sorted_vals = sorted_vals[1:]
        return [rank_value(val, sorted_vals) for val in row]

    return np.matrix([rank_row(row) for row in M])


if __name__ == "__main__":
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
