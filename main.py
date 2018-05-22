import random
import pickle
from scpd.cf import (HTTP_POOL, DiskCodeExtractor, ParticipantExtractor,
                     BatchSubmissionExtractor, CODEFORCES_POOL,
                     get_cached_rated_list, FilterProvider)
from scpd.source import SourceCodePairing
from scpd.features.base import (BaseFeatureExtractor,
                                LabelerPairFeatureExtractor)

MAGICAL_SEED = 42

if __name__ == "__main__":
    random.seed(MAGICAL_SEED)

    # participant_ex = ParticipantExtractor(get_cached_rated_list())
    # participants = participant_ex.extract(500)

    # submissions_ex = BatchSubmissionExtractor(
    #     CODEFORCES_POOL, participants, count=10000)
    # submissions = submissions_ex.extract(FilterProvider().filter(), limit=10)
    # with open("submissions.dat", "wb") as f:
    #     pickle.dump(submissions, f)

    submissions = pickle.load(open("submissions.dat", "rb"))
    code_ex = DiskCodeExtractor(HTTP_POOL, submissions)
    sources = code_ex.extract()

    random.seed(MAGICAL_SEED*2)
    pair_maker = SourceCodePairing()
    paired_sources = pair_maker.make_pairs(sources, k1=10000, k2=10000)

    feature_extractor = LabelerPairFeatureExtractor(BaseFeatureExtractor())
    
    feature_extractor.extract(paired_sources)
