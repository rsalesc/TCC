from contextlib import contextmanager

from . import helpers as fh
from ..source import prefetch
from .extractor import FeatureExtractor, BatchFeatureExtractor


class HelperBasedFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def helpers(self):
        raise NotImplementedError()

    def extract_header(self):
        header = []
        for helper in self.helpers():
            header.extend(helper.__features__)
        return header

    def extract_row(self, source):
        res = []
        for helper in self.helpers():
            res.extend(helper(source))
        return res


class BaseFeatureExtractor(HelperBasedFeatureExtractor):
    def __init__(self):
        super().__init__()

    def helpers(self):
        return [
            fh.author, fh.line_length_statistics, fh.params_length_statistics,
            fh.log_tabs, fh.log_spaces, fh.whitespace_ratio,
            fh.tab_indent_ratio, fh.log_empty_lines, fh.brace_same_line_ratio,
            fh.log_functions, fh.log_decls, fh.log_ternaries,
            fh.compound_statements, fh.tokens, fh.log_unique_keywords
        ]


class PairFeatureExtractor(FeatureExtractor):
    def __init__(self, extractor):
        super().__init__()
        self._extractor = extractor

    def _append_id(self, seq, identifier):
        return list(map(lambda x: "%s_%s" % (x, str(identifier)), seq))

    def extract_header(self):
        extracted_header = self._extractor.extract_header()
        header = self._append_id(extracted_header, 1)
        header.extend(self._append_id(extracted_header, 2))
        return header

    def extract_row(self, sources):
        if not isinstance(sources, tuple) or len(sources) != 2:
            raise AssertionError("sources must be a 2-length tuple")
        a, b = sources
        row = self._extractor.extract_row(a)
        row.extend(self._extractor.extract_row(b))
        return row


class SmartPairFeatureExtractor(PairFeatureExtractor):
    def __init__(self, extractor):
        super().__init__(extractor)

    def extract_into_frame(self, sources):
        unique_set = {}
        for a, b in sources:
            unique_set[id(a)] = a
            unique_set[id(b)] = b

        unique_sources = []
        for source in unique_set.values():
            unique_sources.append(source)
        _, ex_rows = self._extractor.extract_into_frame(unique_sources)

        row_map = {}
        for i, row in enumerate(ex_rows):
            row_map[id(unique_sources[i])] = row
        rows = []
        for a, b in sources:
            row = []
            row.extend(row_map[id(a)])
            row.extend(row_map[id(b)])
            rows.append(row)

        return self.extract_header(), rows


class LabelerPairFeatureExtractor(FeatureExtractor):
    def __init__(self, extractor):
        super().__init__()
        self._extractor = extractor

    def extract_header(self):
        header = self._extractor.extract_header()
        header.append("label")
        return header

    def extract_row(self, sources):
        row = self._extractor.extract_row(sources)
        row.append(0 if sources[0].author() == sources[1].author() else 1)
        return row

    def extract_into_frame(self, sources):
        header, rows = self._extractor.extract_into_frame(sources)
        header.append("label")
        for i, row in enumerate(rows):
            row.append(0 if sources[i][0].author() == sources[i][1].author()
                       else 1)
        return header, rows


class PrefetchBatchFeatureExtractor(BatchFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @contextmanager
    def bootstrap(self, objs):
        filtered_objs = map(lambda x: (x, ) if not isinstance(x, tuple) else x,
                            objs)
        flattened_objs = sum(filtered_objs, ())
        with prefetch(flattened_objs) as _:
            yield objs
