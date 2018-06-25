from contextlib import contextmanager

from . import helpers as fh
from ..source import prefetch
from .extractor import FeatureExtractor, BatchFeatureExtractor


class BaseFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def helpers(self):
        return [
            fh.line_length_statistics,
            fh.params_length_statistics,
            fh.log_tabs,
            fh.log_spaces,
            fh.whitespace_ratio,
            fh.tab_indent_ratio,
            fh.log_empty_lines,
            fh.brace_same_line_ratio,
            fh.log_functions,
            fh.log_decls
        ]

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


class LabelerPairFeatureExtractor(PairFeatureExtractor):
    def __init__(self, extractor):
        super().__init__(extractor)

    def extract_header(self):
        header = PairFeatureExtractor.extract_header(self)
        header.append("label")
        return header

    def extract_row(self, sources):
        row = PairFeatureExtractor.extract_row(self, sources)
        row.append(0 if sources[0].author() == sources[1].author() else 1)
        return row


class PrefetchBatchFeatureExtractor(BatchFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @contextmanager
    def bootstrap(self, objs):
        filtered_objs = map(lambda x: (x) if not isinstance(x, tuple) else x,
                            objs)
        flattened_objs = sum(filtered_objs, ())
        with prefetch(flattened_objs) as _:
            #print(len(list(filter(lambda x: x._ast is None, flattened_objs))))
            yield objs
