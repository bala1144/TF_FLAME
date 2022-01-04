
import time

class Stopwatch(object):
    """A stopwatch utility for timing execution that can be used as a regular
    object or as a context manager.
    NOTE: This should not be used an accurate benchmark of Python code, but a
    way to check how much time has elapsed between actions. And this does not
    account for changes or blips in the system clock.
    Instance attributes:
    start_time -- timestamp when the timer started
    stop_time -- timestamp when the timer stopped
    As a regular object:
    >>> stopwatch = Stopwatch()
    >>> stopwatch.start()
    >>> time.sleep(1)
    >>> 1 <= stopwatch.time_elapsed <= 2
    True
    >>> time.sleep(1)
    >>> stopwatch.stop()
    >>> 2 <= stopwatch.total_run_time
    True
    As a context manager:
    >>> with Stopwatch() as stopwatch:
    ...     time.sleep(1)
    ...     print repr(1 <= stopwatch.time_elapsed <= 2)
    ...     time.sleep(1)
    True
    >>> 2 <= stopwatch.total_run_time
    True
    """

    def __init__(self, task=None):
        """Initialize a new `Stopwatch`, but do not start timing."""
        self.start_time = None
        self.stop_time = None
        self.task = task

    def start(self):
        """Start timing."""
        self.start_time = time.time()


    def __enter__(self):
        """Start timing and return this `Stopwatch` instance."""
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        """Stop timing.
        If there was an exception inside the `with` block, re-raise it.
        >>> with Stopwatch() as stopwatch:
        ...     raise Exception
        Traceback (most recent call last):
            ...
        Exception
        """
        elapsed_time = time.time() - self.start_time

        if self.task is not None:
            print('Elapsed time for ', self.task, elapsed_time)
        else:
            print('Elapsed time for ', elapsed_time)