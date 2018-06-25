"""Module responsible for extracting CF data."""
import os
import json
import pickle
import random
import time
import concurrent.futures as concur
from tqdm import tqdm
from requests import Request, Session
from bs4 import BeautifulSoup

from . import source
from . import utils
from .req import Throttler, RequestsPool, send_request, json_send_request

MAGICAL_SEED = 42
CLE_TIMEOUT = 1
SUBMISSION_COUNT = 10000
CURRENT_DIR = os.path.abspath(os.getcwd())
CODEFORCES_URL = 'http://codeforces.com'
CODEFORCES_API_URL = 'http://codeforces.com/api'
CACHE_PATH = os.path.join(CURRENT_DIR, ".cache/cf")
RATED_LIST_PATH = os.path.join(CACHE_PATH, "ratedList.json")
DESKTOP_AGENTS = [
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'
]


def get_csrf_token(s):
    soup = BeautifulSoup(s.get(CODEFORCES_URL).content, "lxml")
    meta = soup.find("meta", dict(name='X-Csrf-Token'))
    if meta is None:
        raise AssertionError(
            "Couldn't find X-Csrf-Token in Codeforces main page.")
    token = meta['content']
    return token


def get_cached_json(names):
    if not isinstance(names, list):
        names = [names]
    if len(names) > 0:
        while len(names) > 0 and names[0] == '/':
            if len(names[0]) == 0:
                names = names[1:]
                continue
            names[0] = names[0][1:]
    return os.path.join(CACHE_PATH, *names)


def get_api_url(endpoint):
    return '%s/%s' % (CODEFORCES_API_URL, endpoint)


def check_from_api(r):
    if r is None:
        raise IOError("Codeforces API request failed before")
    r = r.json()
    if r["status"] != "OK":
        raise IOError("Codeforces API failed")
    return r["result"]


def get_from_api(req):
    r = send_request(req)
    return check_from_api(r)


def cf_send_request(req, session=None):
    while True:
        try:
            r = send_request(req, session)
            check_from_api(r)
            return r
        except IOError:
            time.sleep(CLE_TIMEOUT)


def get_rated_list():
    req = Request(
        'GET',
        url=get_api_url("user.ratedList"),
        params={'activeOnly': 'true'})
    return get_from_api(req)


def get_cached_rated_list(force=False):
    if not force and os.path.isfile(RATED_LIST_PATH):
        return json.load(open(RATED_LIST_PATH))
    res = get_rated_list()
    json.dump(res, utils.opens(RATED_LIST_PATH, 'w'))
    return res


def printer(resp):
    print(resp)


CODEFORCES_POOL = RequestsPool(
    4, throttler=Throttler(3.5), session=Session(), sender=cf_send_request)
HTTP_POOL = RequestsPool(
    2, throttler=Throttler(1), session=Session(), sender=json_send_request)
PROCESSING_POOL = concur.ThreadPoolExecutor(max_workers=4)
CSRF_TOKEN = get_csrf_token(HTTP_POOL._session)


class ParticipantExtractor():
    def __init__(self, rated_list):
        self._n = len(rated_list)
        self._rated_list = rated_list

    def extract(self, K, props=None):
        if props is None:
            return utils.sample_list(self._rated_list, K)
        else:
            raise NotImplementedError()


class Problem():
    def __init__(self, problem):
        self.contest_id = problem.get("contestId")
        self.index = problem.get("index")
        self.name = problem["name"]


class Submission():
    def __init__(self, submission):
        members = submission["author"]["members"]
        self.authors = list(map(lambda x: x["handle"], members))
        self.time = submission["creationTimeSeconds"]
        self.id = submission["id"]
        self.type = submission["author"]["participantType"]
        self.problem = Problem(submission["problem"])

    def author(self):
        if len(self.authors) == 0:
            return None
        return self.authors[0]

    def code_request(self):
        data = {'submissionId': self.id, 'csrf_token': CSRF_TOKEN}
        headers = {'User-Agent': random.choice(DESKTOP_AGENTS)}
        return Request(
            'POST',
            url="%s/data/submitSource" % CODEFORCES_URL,
            data=data,
            headers=headers)


class CodeforcesSource(source.SourceCode):
    def __init__(self, submission, code=None, path=None):
        source.SourceCode.__init__(
            self, submission.author(), code=code, path=path)
        self._submission = submission

    def submission(self):
        return self._submission


class SubmissionExtractor():
    def __init__(self, pool, handle, count=SUBMISSION_COUNT):
        self._pool = pool
        self._handle = handle
        self._count = count

    def extract(self):
        req_params = {'handle': self._handle, 'count': self._count, 'from': 1}
        req = Request('GET', url=get_api_url('user.status'), params=req_params)
        return self._pool.submit(req)


class BatchSubmissionExtractor():
    def __init__(self,
                 pool,
                 participants,
                 count=SUBMISSION_COUNT,
                 monitor=True):
        self._pool = pool
        self._participants = participants
        self._count = count
        self._monitor = monitor

    def extract(self, filter_fn=lambda x: True, limit=None):
        futures = []
        for participant in self._participants:
            handle = participant["handle"]
            extractor = SubmissionExtractor(
                self._pool, handle, count=self._count)
            futures.append(extractor.extract())

        submissions = []
        as_completed = concur.as_completed(futures)
        if self._monitor:
            as_completed = tqdm(
                as_completed, total=len(futures), desc='Submission Extraction')
        failed = 0
        for future in as_completed:
            try:
                r = check_from_api(future.result(timeout=5))
            except json.JSONDecodeError:
                failed += 1
                continue
            r = list(filter(filter_fn, r))
            if limit is not None:
                r = r[:limit]
            for submission in r:
                submissions.append(Submission(submission))
        print("Submissions from %d participants skipped..." % failed)

        return submissions


class DiskCodeExtractor():
    def __init__(self, pool, submissions, monitor=True, download=True):
        self._pool = pool
        self._submissions = submissions
        self._monitor = monitor
        self._download = download

    def extract(self):
        futures = []
        for submission in self._submissions:
            output_path = get_cached_json(str(submission.id))
            if os.path.isfile(output_path) or not self._download:
                futures.append(None)
            else:
                futures.append(self._pool.submit(submission.code_request()))

        fulfilled = 0
        new_requests = len(list(filter(lambda x: x is not None, futures)))
        sources = []
        enumerated = enumerate(futures)
        if self._monitor:
            enumerated = tqdm(
                enumerated, total=len(futures), desc='Code Extraction')
        for i, future in enumerated:
            # try/catch?
            if self._monitor:
                enumerated.set_postfix(
                    skipped="{}/{}".format(fulfilled, new_requests))
            submission = self._submissions[i]
            output_path = get_cached_json(str(submission.id))
            if future is not None:
                try:
                    r = future.result(timeout=5)
                    r = r.json()
                    with utils.opens(output_path, "w") as f:
                        f.write(r["source"])
                    del future
                except (IOError, json.JSONDecodeError) as e:
                    pass
            if os.path.isfile(output_path):
                sources.append(CodeforcesSource(submission, path=output_path))
                if future is not None:
                    fulfilled += 1
        if self._monitor:
            print("Skipped {} entries...".format(len(futures) - len(sources)))
        return sources


class FilterProvider():
    def __init__(self):
        self._seen = {}

    def filter(self):
        def filter_fn(r):
            if r["id"] in self._seen:
                return False
            if r["verdict"] != "OK":
                return False
            if r["author"]["participantType"] != "CONTESTANT":
                return False
            if r["author"]["ghost"]:
                return False
            if "contestId" not in r or r["contestId"] > 10000:
                return False
            if "C++" not in r["programmingLanguage"]:
                return False
            self._seen[r["id"]] = True
            return True

        return filter_fn
