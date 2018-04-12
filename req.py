import time
import threading

ONE_SECOND = 1
TIMEOUT = 10

class Throttler:
    def __init__(self, qps):
        self._qps = qps
        self._requests = {}
        self._pending_removal = []
        self._pending_lock = threading.Lock()
        self._qps_lock = threading.Lock()

    def _cleanup(self, tempo):
        with self._pending_lock:
            now_pending = []
            for pending in self._pending_removal:
                if pending in self._requests:
                    time = self._requests[pending]
                    if time + ONE_SECOND >= tempo:
                        now_pending.append(pending)
                    else:
                        del self._requests[pending]
            self._pending_removal = now_pending

    def _check_lock(self):
        self._qps_lock.acquire(blocking=False)
        with self._pending_lock:
            if len(self._requests) < self._qps:
                self._qps_lock.release()
    
    def throttle(self, id):
        tempo = time.time()
        self._cleanup(tempo)
        self._check_lock()

        with self._pending_lock:
            tempo = time.time()
            self._requests[id] = tempo

    def finish(self, id):
        self._pending_removal.append(id)
        self._cleanup(time.time())
        self._check_lock()

def send_request(req, session=None):
    if session is not None:
        return session.send(req.prepare(), timeout=TIMEOUT)
    else:
        raise "ETA"

class ThreadedRequest(threading.Thread):
    def __init__(self, req, fn=None, session=None):
        threading.Thread.__init__(self)
        self._request = req
        self._fn = fn
        self._session = session

    def run(self):
        self._fn(send_request(self._request, self._session))

class RequestsPool:
    def __init__(self, size, throttler=None, session=None):
        self._size = size
        self._semaphore = threading.BoundedSemaphore(size)
        self._throttler = throttler
        self._counter_lock = threading.Lock()
        self._counter = 0
        self._session = session

    def _get_id(self):
        with self._counter_lock:
            res = self._counter
            self._counter += 1
            return res

    def submit(self, req, fn):
        self._semaphore.acquire()
        tid = self._get_id()
        if self._throttler is not None:
            self._throttler.throttle(tid)
        
        def release(*args, **kwargs):
            if self._throttler is not None:
                self._throttler.finish(tid)
            self._semaphore.release()
            fn(*args, **kwargs)
        
        ThreadedRequest(req, release, self._session).start()

    def fetch(self, req):
        with self._semaphore:
            tid = self._get_id()
            if self._throttler is not None:
                self._throttler.throttle(tid)

            resp = send_request(req, self._session)
            if self._throttler is not None:
                self._throttler.finish(tid)
            
            return resp