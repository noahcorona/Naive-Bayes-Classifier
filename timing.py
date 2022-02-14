import time


# Decorator to time functions
def timeit(func):
    def timed(*args, **kw):
        print('\nStarting function %r' % func.__name__)
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', func.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r completed in %2.2f ms\n' % (func.__name__, (te - ts) * 1000))
        return result

    return timed
