from ..source import SourceCode


class JoernSourceCode(SourceCode):
    def __init__(self, source, ast):
        super().__init__(self, source._author, source._code, source._path)
        self._ast = ast

    def ast(self):
        return self._ast
