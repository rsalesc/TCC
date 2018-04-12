from requests import Request, Session
from req import Throttler, RequestsPool

def printer(resp):
    print(resp)

if __name__ == "__main__":
    throttler = Throttler(3)
    pool = RequestsPool(3, throttler, Session())
    printer(pool.fetch(Request('GET', url='http://codeforces.com')))