import traceback
from contextlib import contextmanager
from pygments.lexers.c_cpp import CppLexer


def get_unique_objects(objs):
    unique = {}
    for obj in objs:
        unique[id(obj)] = obj

    return unique.values()


@contextmanager
def prefetch(sources):
    unique_sources = get_unique_objects(sources)
    try:
        for source in unique_sources:
            source.prefetch()
        yield sources
    except Exception as e:
        print(traceback.format_exc())
    finally:
        for source in unique_sources:
            source.unfetch()


class SourceCode():
    def __init__(self, author, code=None, path=None):
        self._author = author
        self._code = code
        self._path = path
        self._tokens = None

    def author(self):
        return self._author

    def path(self):
        return self._path

    def fetch(self):
        if self._code is not None:
            return self._code
        elif self._path is not None:
            try:
                return open(self._path, encoding="utf-8").read()
            except UnicodeDecodeError:
                return open(self._path, encoding="iso-8859-1").read()
        else:
            raise AssertionError('No source code')

    def prefetch(self):
        if self._code is None:
            self._code = self.fetch()
        self.prefetch_tokens()

    def unfetch(self):
        if self._code is not None and self._path is not None:
            self._code = None
        self.unfetch_tokens()

    def fetch_tokens(self):
        if self._tokens is not None:
            return self._tokens
        code = self.fetch()
        lexer = CppLexer()
        return list(lexer.get_tokens_unprocessed(code))

    def prefetch_tokens(self):
        if self._tokens is None:
            self._tokens = self.fetch_tokens()
        return self._tokens

    def unfetch_tokens(self):
        self._tokens = None

    def __str__(self):
        sec = self._code if self._code is not None else self._path
        return '(%s, %s)' % (self._author, sec)
