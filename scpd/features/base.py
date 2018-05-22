from . import helpers as fh
from .extractor import FeatureExtractor


class BaseFeatureExtractor(FeatureExtractor):
    def __init__(self):
        FeatureExtractor.__init__(self)

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
