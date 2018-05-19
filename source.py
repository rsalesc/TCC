class SourceCode:
    def __init__(self, author, code=None, path=None):
        self._author = author
        self._code = code
        self._path = path

    def author(self):
        return self._author
    
    def fetch(self):
        if self._code is not None:
            return self._code
        elif self._path is not None:
            return open(self._path).read()
        else:
            raise AssertionError("No source code")
