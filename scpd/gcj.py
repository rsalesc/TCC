import os
import json

from . import utils, source


CURRENT_DIR = os.path.abspath(os.getcwd())
CACHE_PATH = os.path.join(CURRENT_DIR, ".cache/gcj")
METADATA_PATH = os.path.join(CACHE_PATH, "CodeJamMetadata.json")
STATS_PATH = os.path.join(CACHE_PATH, "stats")
CODE_PATH = os.path.join(CACHE_PATH, "codejamfolder")
USERS_PATH = os.path.join(CACHE_PATH, "users")


def get_code_path(problem, username, folder):
    fn = "p_%s.%s" % (str(problem), str(username))
    return os.path.join(CODE_PATH, folder, username, fn)


def get_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


def get_year_stats(year, lang):
    fn = os.path.join(STATS_PATH, lang, "year", "{}.json".format(year))
    with open(fn) as f:
        return json.load(f)


def get_round_stats(round_id, lang):
    fn = os.path.join(STATS_PATH, lang, "{}.json".format(round_id))
    with open(fn) as f:
        return json.load(f)


def get_round_users(round_id, lang):
    """Returns pair (user, solved) for round_id, with language lang."""
    return list(get_round_stats(round_id, lang)["users"].items())


def get_year_users(year, lang):
    """Returns pair (user, solved) for year, with language lang."""
    return list(get_year_stats(year, lang)["users"].items())


def get_round_problems(round_id, lang):
    """Returns pair (problem, solved) for round_id, with language lang."""
    return list(get_round_stats(round_id, lang)["problems"].items())


def get_year_problems(year, lang):
    """Returns pair (problem, solved) for year, with language lang."""
    return list(get_year_stats(year, lang)["problems"].items())


def get_user_stats_for_round(user, round_id, lang):
    problems = get_round_problems(round_id, lang)

    res = {}
    for problem, _ in problems:
        fn = get_code_path(problem, user, lang)
        if os.path.isfile(fn):
            res[problem] = 1

    return res


def get_user_stats_for_year(user, year, lang):
    problems = get_year_problems(year, lang)

    res = {}
    for problem, _ in problems:
        fn = get_code_path(problem, user, lang)
        if os.path.isfile(fn):
            res[problem] = 1

    return res


class CodejamSource(source.SourceCode):
    def __init__(self, submission):
        super().__init__(
            submission.author(), code=None, path=submission.code_path())
        self._submission = submission

    def submission(self):
        return self._submission


class Submission:
    def __init__(self, user, problem, lang):
        self._user = user
        self._problem = problem
        self._language = lang

    def author(self):
        return self._user

    def problem(self):
        return self._problem

    def language(self):
        return self._language

    def code_path(self):
        return get_code_path(self._problem, self._user, self._language)

    def get_source(self):
        return CodejamSource(self)


class RandomParticipantExtractor:
    def __init__(self, years, lang):
        self._years = years
        self._problems = []
        self._users = []
        for year in years:
            self._problems.extend(get_year_problems(year, lang))
            self._users.extend(get_year_users(year, lang))

    def extract(self, K, at_least=1):
        users = filter(lambda x: x[1] >= at_least, self._users)
        users, _ = zip(*users)

        if sum(K) > len(users):
            raise AssertionError(
                "There are not enough users that solved `at_least` problems")

        samples = utils.sample_list(users, sum(K))
        acc = 0
        res = []
        for k in K:
            res.append(samples[acc:acc+k])
            acc += k

        return res


class RandomSubmissionExtractor:
    def __init__(self, years, lang):
        self._years = years
        self._problems = []
        self._users = []
        self._language = lang
        for year in years:
            self._problems.extend(get_year_problems(year, lang))
            self._users.extend(get_year_users(year, lang))

    def extract(self, users, limit):
        res = []
        for user in users:
            solved = {}
            for year in self._years:
                solved.update(get_user_stats_for_year(
                    user, year, self._language))
            problems = utils.sample_list(
                list(solved.keys()), min(limit, len(solved)))
            res.extend([Submission(user, problem, self._language)
                        for problem in problems])

        return res
