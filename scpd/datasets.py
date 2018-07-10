import os
import pickle
from abc import ABCMeta, abstractmethod

from . import utils
from .cf import (ParticipantExtractor, get_cached_rated_list,
                 BatchSubmissionExtractor, CODEFORCES_POOL, FilterProvider,
                 DiskCodeExtractor, HTTP_POOL, PROCESSING_POOL)
from .joern.parser import BatchJoernParser


class DatasetBuilder(metaclass=ABCMeta):
    @abstractmethod
    def extract(self, force=False):
        """Returns a set of SourceCode training and test objects."""
        raise NotImplementedError()


def extract_cf_codes(submissions, download):
    # maybe add force flag?
    code_extractor = DiskCodeExtractor(
        HTTP_POOL, submissions, download=download)
    return code_extractor.extract()


def extract_joern_codes(sources, force=False):
    parser = BatchJoernParser(PROCESSING_POOL, sources)
    return parser.parse(force=force)


class CodeforcesDatasetBuilder():
    SUBMISSION_API_COUNT = 10000

    def __init__(self,
                 training_size,
                 test_size,
                 training_file,
                 test_file,
                 submissions_per_user,
                 training_only=False,
                 download=True):
        self._participant_extractor = ParticipantExtractor(
            get_cached_rated_list())
        self._training_size = training_size
        self._test_size = test_size
        self._training_file = training_file
        self._test_file = test_file
        self._submissions_per_user = submissions_per_user
        self._download = download
        self._training_only = training_only

    def fetch_dataset(self, K):
        participants = self._participant_extractor.extract(K)
        submission_extractor = BatchSubmissionExtractor(
            CODEFORCES_POOL,
            participants,
            count=CodeforcesDatasetBuilder.SUBMISSION_API_COUNT)
        return submission_extractor.extract(
            FilterProvider().filter(), limit=self._submissions_per_user)

    def extract_submissions(self, force=False):
        should_fetch = (force or not os.path.isfile(self._training_file)
                        or (not self._training_only and not os.path.isfile(self._test_file)))
        training_submissions = None
        test_submissions = None

        if should_fetch:
            training_submissions = self.fetch_dataset(self._training_size)
            with utils.opens(self._training_file, "wb") as f:
                pickle.dump(training_submissions, f)

            if not self._training_only:
                test_submissions = self.fetch_dataset(self._test_size)
                with utils.opens(self._test_file, "wb") as f:
                    pickle.dump(test_submissions, f)

        if training_submissions is None:
            training_submissions = pickle.load(
                utils.opens(self._training_file, "rb", encoding=None))
        if test_submissions is None and not self._training_only:
            test_submissions = pickle.load(
                utils.opens(self._test_file, "rb", encoding=None))

        return training_submissions, test_submissions

    def extract_raw(self, force=False):
        training_submissions, test_submissions = self.extract_submissions(
            force)

        if test_submissions is None:
            return extract_cf_codes(training_submissions, self._download), None

        training_sources, test_sources = extract_cf_codes(
            training_submissions, self._download), extract_cf_codes(
                test_submissions, self._download)

        return training_sources, test_sources

    def extract(self, force=False, force_joern=False):
        training_sources, test_sources = self.extract_raw(force=force)

        if test_sources is None:
            return extract_joern_codes(training_sources, force_joern)

        return extract_joern_codes(training_sources,
                                   force_joern), extract_joern_codes(
                                       test_sources, force_joern)
