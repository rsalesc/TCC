import os
import pickle
from abc import ABCMeta, abstractmethod

from . import utils, gcj
from .cf import (ParticipantExtractor, get_cached_rated_list,
                 BatchSubmissionExtractor, CODEFORCES_POOL, FilterProvider,
                 DiskCodeExtractor, HTTP_POOL, PROCESSING_POOL)
from .joern.parser import BatchJoernParser
from .caide.optimizer import CodeOptimizer, BatchSourceOptimizer


def extract_cf_codes(submissions, download):
    # maybe add force flag?
    code_extractor = DiskCodeExtractor(
        HTTP_POOL, submissions, download=download)
    return code_extractor.extract()


def extract_joern_codes(sources, force=False):
    parser = BatchJoernParser(PROCESSING_POOL, sources)
    return parser.parse(force=force)


def gcj_codes_plugin(builder, descriptor, input, force, verbose):
    return [sub.get_source() for sub in input]


def cf_codes_plugin(builder, descriptor, input, force, verbose):
    return extract_cf_codes(input, builder._download)


def cf_joern_plugin(builder, descriptor, input, force, verbose):
    return extract_joern_codes(input, force=force)


def cf_caide_plugin(includes, *args, **kwargs):
    optimizer = CodeOptimizer(includes, *args, **kwargs)
    batch = BatchSourceOptimizer(PROCESSING_POOL, optimizer, **kwargs)

    def cf_caide_plugin(builder, descriptor, input, force, verbose):
        if descriptor["caide"] is not False:
            batch.run(input, force)
        return input

    return cf_caide_plugin


def joern_plugin(*args, **kwargs):
    return cf_joern_plugin(*args, **kwargs)


def caide_plugin(*args, **kwargs):
    return cf_caide_plugin(*args, **kwargs)


class CodeforcesDescriptor:
    def __init__(self, name, participants, submissions_per_participant,
                 **kwargs):
        self.name = name
        self.participants = participants
        self.submissions_per_participant = submissions_per_participant
        self._kwargs = kwargs

    def __getitem__(self, key):
        if key not in self._kwargs:
            return None
        return self._kwargs[key]

    def path(self):
        return self["path"]


class CodejamDescriptor(CodeforcesDescriptor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DatasetBuilder:
    def __init__(self, descriptors, plugins=[]):
        self._plugins = list(plugins)
        self._descriptors = descriptors

    def plugins(self):
        return self._plugins

    def descriptors(self):
        return self._descriptors

    def fetch_datasets(self):
        pass

    def check_cache_for_datasets(self):
        are_cached = 0
        for descriptor in self._descriptors:
            path = descriptor.path()
            if path is not None:
                path = os.path.abspath(path)
                if os.path.isfile(path):
                    are_cached += 1

        if are_cached == len(self._descriptors):
            res = []
            for descriptor in self._descriptors:
                path = os.path.abspath(descriptor.path())
                dataset = pickle.load(utils.opens(path, "rb", encoding=None))
                assert isinstance(dataset, list)
                res.append(dataset)
            return res
        elif are_cached > 0:
            raise AssertionError("have only partial datasets")

        return None

    def cache_descriptor(self, descriptor, dataset):
        if descriptor.path() is not None:
            path = os.path.abspath(descriptor.path())
            with utils.opens(path, "wb") as f:
                pickle.dump(dataset, f)

    def extract(self, force=False, verbose=True):
        datasets = self.fetch_datasets()
        res = []
        for i, dataset in enumerate(datasets):
            descriptor = self._descriptors[i]
            if verbose:
                print("Processing descriptor {}...".format(descriptor.name))
            for plugin in self._plugins:
                dataset = plugin(self, descriptor, dataset, force, verbose)
            res.append(dataset)
        return res


class CodejamDisjointDatasetBuilder(DatasetBuilder):
    def __init__(self, descriptors, years, lang, at_least=1, plugins=[]):
        super().__init__(descriptors, plugins=plugins)
        self._years = [str(year) for year in years]
        self._lang = lang
        self._at_least = at_least
        self._participant_extractor = gcj.RandomParticipantExtractor(
            years, lang)
        self._submission_extractor = gcj.RandomSubmissionExtractor(years, lang)

    def get_dataset_sizes(self):
        return list(map(lambda x: x.participants, self._descriptors))

    def fetch_datasets(self):
        cached_datasets = self.check_cache_for_datasets()
        if cached_datasets is not None:
            return cached_datasets

        participants = self._participant_extractor.extract(
            self.get_dataset_sizes(), at_least=self._at_least)

        res = []
        for i, p in enumerate(participants):
            descriptor = self.descriptors()[i]
            dataset = self._submission_extractor.extract(
                p, limit=descriptor.submissions_per_participant)
            self.cache_descriptor(descriptor, dataset)
            res.append(dataset)

        return res


class CodeforcesDisjointDatasetBuilder(DatasetBuilder):
    SUBMISSION_API_COUNT = 10000

    def __init__(self, descriptors, download=True, plugins=[]):
        super().__init__(descriptors, plugins=plugins)
        self._download = download
        self._participant_extractor = ParticipantExtractor(
            get_cached_rated_list())

    def get_dataset_sizes(self):
        return list(map(lambda x: x.participants, self._descriptors))

    def fetch_datasets(self):
        cached_datasets = self.check_cache_for_datasets()
        if cached_datasets is not None:
            return cached_datasets

        participants = self._participant_extractor.extract(
            self.get_dataset_sizes())

        res = []
        for i, p in enumerate(participants):
            descriptor = self._descriptors[i]
            extractor = BatchSubmissionExtractor(
                CODEFORCES_POOL,
                p,
                count=CodeforcesDisjointDatasetBuilder.SUBMISSION_API_COUNT)
            dataset = extractor.extract(
                FilterProvider().filter(),
                limit=descriptor.submissions_per_participant)
            self.cache_descriptor(descriptor, dataset)
            res.append(dataset)
        return res


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
                        or (not self._training_only
                            and not os.path.isfile(self._test_file)))
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
