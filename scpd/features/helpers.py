import math
import numpy as np
import pygments.token as pygtoken
import string
from functools import wraps

from ..graph.visitor import TreeVisitor
from ..joern import types as jt
from ..utils import isiterable

WHITESPACE_CHARS = string.whitespace
INDENT_CHARS = " \t"
CONTROL_TOKENS = {
    "do": True,
    "else": True,
    "if": True,
    "switch": True,
    "while": True,
    "for": True
}


def feature(*args, **kwargs):
    if len(args) == 0:
        raise AssertionError("Feature should have a name.")

    name_args_len = len(args)

    def feature_inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            # make sure the result is a tuple of features
            if not isiterable(res):
                res = (res, )
            res = tuple(res)
            if len(res) != name_args_len:
                raise AssertionError(
                    "Helper return length should have same length of feature names."
                )
            return res

        wrapper.__features__ = tuple(args)
        return wrapper

    return feature_inner


def unnamed_feature(func):
    return feature(func.__name__)(func)


@feature("avg_line_length", "stddev_line_length")
def line_length_statistics(source):
    """Avg and stddev of line length across the code."""
    code = source.fetch()
    lines = list(code.splitlines())
    avg = 0
    stddev = 0
    for line in lines:
        avg += len(line) / len(lines)
    for line in lines:
        stddev += (len(line) - avg)**2 / len(lines)
    stddev = math.sqrt(stddev)
    return avg, stddev


@feature("avg_params_length", "stddev_params_length")
def params_length_statistics(source):
    """Get avg and stddev of paramater list length across the code."""
    ast_root = source.fetch_ast()
    visitor = ParamsLengthVisitor()
    visitor.visit(ast_root)
    return visitor.result()


@unnamed_feature
def log_tabs(source):
    """Get ln(no_of_tabs/len(code) + 1)."""
    code = source.fetch()
    return math.log(code.count("\t") / len(code) + 1)


@unnamed_feature
def log_spaces(source):
    """Get ln(no_of_spaces/len(code) + 1)."""
    code = source.fetch()
    return math.log(code.count(" ") / len(code) + 1)


@unnamed_feature
def whitespace_ratio(source):
    code = source.fetch()
    whitespaces = sum(map(code.count, WHITESPACE_CHARS))
    if whitespaces == len(code):
        return 1.0
    return whitespaces / (len(code) - whitespaces)


@unnamed_feature
def tab_indent_ratio(source):
    """Get ratio between tab-indented lines and overall indented lines."""
    code = source.fetch()
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


@unnamed_feature
def log_empty_lines(source):
    """Get ln(empty_lines / len(code) + 1)."""
    code = source.fetch()
    lines = code.splitlines()
    empty_lines = 0
    for line in lines:
        if len(line.strip()) == 0:
            empty_lines += 1
    return math.log(empty_lines / len(code) + 1)


@unnamed_feature
def brace_same_line_ratio(source):
    """Get ratio between same-line open braces and compound statements."""
    ast_root = source.fetch_ast()
    visitor = OpenBraceVisitor()
    visitor.visit(ast_root)
    return visitor.result()


@unnamed_feature
def log_functions(source):
    """Get ln(functions / len(code) + 1)."""
    length = len(source.fetch())
    ast_root = source.fetch_ast()
    visitor = CountVisitor([jt.FUNCTION_DEF])
    visitor.visit(ast_root)
    return math.log(visitor.result() / length + 1)


@unnamed_feature
def log_decls(source):
    """Get ln(declarations / len(code) + 1)."""
    length = len(source.fetch())
    ast_root = source.fetch_ast()
    visitor = CountVisitor([jt.IDENTIFIER_DECL])
    visitor.visit(ast_root)
    return math.log(visitor.result() / length + 1)


@unnamed_feature
def log_ternaries(source):
    """Get ln(ternary_operators / len(code) + 1)."""
    length = len(source.fetch())
    ast_root = source.fetch_ast()
    visitor = CountVisitor([jt.CONDITIONAL_EXPRESSION])
    visitor.visit(ast_root)
    return math.log(visitor.result() / length + 1)


@feature("compound_statement_max_depth", "compound_statement_branching",
         "compound_statement_dangling")
def compound_statements(source):
    ast_root = source.fetch_ast()
    visitor = CompoundStatementsVisitor()
    visitor.visit(ast_root)
    max_depth, branching, dangling = visitor.result()
    return max_depth, branching, math.log(dangling + 1)


@feature("log_tokens", "log_control_tokens", "log_literals")
def tokens(source):
    length = len(source.fetch())
    tokens = source.fetch_tokens()
    filtered_tokens = list(
        filter(lambda x: x[1] not in pygtoken.Comment, tokens))
    control_tokens = list(
        filter(lambda x: x[1] in pygtoken.Keyword and x[2] in CONTROL_TOKENS,
               tokens))
    literal_tokens = list(filter(lambda x: x[1] in pygtoken.Literal, tokens))
    return (math.log(len(filtered_tokens) / length + 1),
            math.log(len(control_tokens) / length + 1),
            math.log(len(literal_tokens) / length + 1))


@unnamed_feature
def log_unique_keywords(source):
    length = len(source.fetch())
    tokens = source.fetch_tokens()
    unique = {}
    for token in tokens:
        if token[1] in pygtoken.Keyword:
            unique[token[1]] = True
    return math.log(len(unique) / length + 1)


class NodeVisitor(TreeVisitor):
    def __init__(self):
        super().__init__()
        self._sum_depth = {}
        self._freq = {}

    def result(self):
        for key, value in self._freq.items():
            self._sum_depth[key] /= value
        return self._sum_depth.itervalues()


class CountVisitor(TreeVisitor):
    def __init__(self, targets):
        super().__init__()
        self._targets = targets
        self._result = 0

    def result(self):
        return self._result

    def visit(self, element):
        if element.type() in self._targets:
            self._result += 1
        for child in element.children():
            self.step(child)


class OpenBraceVisitor(TreeVisitor):
    def __init__(self):
        super().__init__()
        self._same_line = 0
        self._total = 0

    def result(self):
        if self._total == 0:
            return 1.0
        return self._same_line / self._total

    def visit(self, element):
        if element.type() == jt.FUNCTION_DEF:
            for child in element.children():
                if child.type() == jt.COMPOUND_STATEMENT:
                    self._total += 1
                    if child.location().line == element.location().line:
                        self._same_line += 1
        for child in element.children():
            self.step(child)


class ParamsLengthVisitor(TreeVisitor):
    def __init__(self):
        super().__init__()
        self._lengths = []

    def result(self):
        if len(self._lengths) == 0:
            return 0, 0
        return np.average(self._lengths), np.std(self._lengths)

    def visit(self, element):
        if element.type() == jt.PARAMETER_LIST:
            parameters = 0
            for child in element.children():
                if child.type() == jt.PARAMETER:
                    parameters += 1
            self._lengths.append(parameters)

        for child in element.children():
            self.step(child)


class CompoundStatementsVisitor(TreeVisitor):
    def __init__(self):
        super().__init__()
        self._depth = 0
        self._max_depth = 0
        self._children = [0]
        self._child_sum = 0
        self._nodes = 0
        self._dangling = 0

    def result(self):
        return self._max_depth, self._child_sum / max(self._nodes,
                                                      1), self._dangling

    def visit(self, element):
        if element.type() == jt.COMPOUND_STATEMENT:
            self._depth += 1
            self._max_depth = max(self._max_depth, self._depth)
            self._children[-1] += 1
            self._children.append(0)
            if element.parent() is None or element.parent().type() not in (
                    jt.CONTROL_STATEMENTS + [jt.FUNCTION_DEF]):
                self._dangling += 1
        for child in element.children():
            self.step(child)
        if element.type() == jt.COMPOUND_STATEMENT or element.parent() is None:
            if self._children[-1] > 0:
                self._nodes += 1
                self._child_sum += self._children[-1]
            self._children.pop()
