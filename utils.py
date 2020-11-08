import functools
import time
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
