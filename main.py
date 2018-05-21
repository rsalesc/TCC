import random
import pickle
from scpd.cf import HTTP_POOL, DiskCodeExtractor
from scpd.graph import ast

MAGICAL_SEED = 42

if __name__ == "__main__":
    random.seed(MAGICAL_SEED)

    # participant_ex = ParticipantExtractor(get_cached_rated_list())
    # participants = participant_ex.extract(500)

    # submissions_ex = BatchSubmissionExtractor(CODEFORCES_POOL,
    #        participants, count=10000)
    # submissions = submissions_ex.extract(FilterProvider().filter(), limit=10)
    # with open("submissions.dat", "wb") as f:
    # pickle.dump(submissions, f)

    #submissions = pickle.load(open("submissions.dat", "rb"))
    #code_ex = DiskCodeExtractor(HTTP_POOL, submissions)
    #code_ex.extract()

    print(dir(ast))
