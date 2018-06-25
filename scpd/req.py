"""Provides throttling helpers."""
import time
import threading
import requests
import json
import concurrent.futures as concur

ONE_SECOND = 1
TIMEOUT = 10


class Throttler:
    """Provides throttling functionality."""

    def __init__(self, qps):
        self._qps = qps
        self._pending_lock = threading.Lock()
        self._capacity = qps
        self._last = time.time()

    def throttle(self, consumed=1.0, blocking=True):
        with self._pending_lock:
            cur = time.time()
            time_passed = cur - self._last
            self._capacity += time_passed * self._qps
            if self._capacity > self._qps:
                self._capacity = self._qps
            if self._capacity < consumed:
                if not blocking:
                    return False
                needs = (consumed - self._capacity) / self._qps
                time.sleep(needs)
            self._capacity -= consumed
        return True


def send_request(req, session=None):
    if session is not None:
        return session.send(session.prepare_request(req), timeout=TIMEOUT)
    else:
        return requests.Session().send(req.prepare(), timeout=TIMEOUT)


def json_send_request(req, session=None):
    for _ in range(3):
        try:
            r = send_request(req, session)
            r.json()
            return r
        except (IOError, json.JSONDecodeError):
            time.sleep(ONE_SECOND)
    raise IOError("Exceeded 3 tries")


class RequestsPool:
    def __init__(self, size, throttler=None, session=None,
                 sender=send_request):
        self._throttler = throttler
        self._counter_lock = threading.Lock()
        self._counter = 0
        self._session = session
        self._executor = concur.ThreadPoolExecutor(max_workers=size)
        self._sender = sender

    def submit(self, req, fn=None):
        def do():
            if self._throttler is not None:
                self._throttler.throttle()
            return self._sender(req, self._session)

        def release(future):
            if fn is not None:
                fn(future.result())

        f = self._executor.submit(do)
        f.add_done_callback(release)
        return f

    def fetch(self, req):
        if self._throttler is not None:
            self._throttler.throttle()

        resp = send_request(req, self._session)

        return resp
