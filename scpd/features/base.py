from . import helpers as fh
from .extractor import FeatureExtractor


class BaseFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__(self)

    def extract_header(self):
        return [
            "avg_line_length",
            "stddev_line_length",
            "log_tabs",
            "log_spaces",
            "whitespace_ratio",
            "tab_indent_ratio",
        ]   

    def extract_row(self, source):
        res = []
        code = source.fetch()

        res.extend(fh.line_length_statistics(code))
        res.append(fh.log_tabs(code))
        res.append(fh.log_spaces(code))
        res.append(fh.whitespace_ratio(code))
        res.append(fh.tab_indent_ratio(code))

        return res


class PairFeatureExtractor(FeatureExtractor):
    def __init__(self, extractor):
        super().__init__(self)
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
        super().__init__(self, extractor)

    def extract_header(self):
        header = PairFeatureExtractor.extract_header(self)
        header.append("label")
        return header

    def extract_row(self, sources):
        row = PairFeatureExtractor.extract_row(self, sources)
        row.append(0 if sources[0].author() == sources[1].author() else 1)
        return row
