import math
import string


WHITESPACE_CHARS = string.whitespace
INDENT_CHARS = " \t"


def line_length_statistics(code):
    """Avg and stddev of line length across the code."""
    lines = list(code.splitlines())
    avg = 0
    stddev = 0
    for line in lines:
        avg += len(line) / len(lines)
    for line in lines:
        stddev += (len(line) - avg)**2 / len(lines)
    stddev = math.sqrt(stddev)
    return avg, stddev


def log_tabs(code):
    """Get ln(no_of_tabs/len(code) + 1)."""
    return math.log(code.count("\t") + 1)


def log_spaces(code):
    """Get ln(no_of_spaces/len(code) + 1)."""
    return math.log(code.count("\n") + 1)


def whitespace_ratio(code):
    whitespaces = sum(map(code.count, WHITESPACE_CHARS))
    if whitespaces == len(code):
        return 1.0
    return whitespaces / (len(code) - whitespaces)


def tab_indent_ratio(code):
    """Get ratio between tab-indented lines and overall indented lines."""
    lines = code.splitlines()
    indented_lines = 0
    tab_indented_lines = 0
    for line in lines:
        if line.startswith(tuple(INDENT_CHARS)):
            indented_lines += 1
            if line.startswith("\t"):
                tab_indented_lines += 1
    if indented_lines == 0:
        return 0.5
    return tab_indented_lines / indented_lines
