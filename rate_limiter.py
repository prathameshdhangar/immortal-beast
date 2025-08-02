import time


class RateLimiter:

    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def can_call(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        return len(self.calls) < self.max_calls

    def record_call(self):
        self.calls.append(time.time())
