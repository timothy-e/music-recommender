import functools
import time

def timeit(func):
    @functools.wraps(func)
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()

        print(f"{func.__name__:<20} {f'{(te-ts) * 100:,.2f} ms':>10}")
        return result
    return timed