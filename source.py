class SourceCode:
    def __init__(self, author, code):
        self._author = author
        self._code = code

    def get_author(self):
        return self._author
    
    def get_code(self):
        return self._code