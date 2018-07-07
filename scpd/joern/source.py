import pickle

from ..source import SourceCode
from ..graph.graph import CsvGraphParser
from ..graph.astree import to_ast

ID_FIELD = "key"
NODES_CACHE = "{}.n.ast"
EDGES_CACHE = "{}.e.ast"


def check_joern_data_from(nodes_csv, edges_csv):
    fn = open(nodes_csv, "r", encoding="utf-8")
    fe = open(edges_csv, "r", encoding="utf-8")
    csv_parser = CsvGraphParser(fn, fe, id_field=ID_FIELD)
    csv_parser.check()
    csv_parser.cleanup()


def fetch_joern_data_from(nodes_csv, edges_csv):
    fn = open(nodes_csv, "r", encoding="utf-8")
    fe = open(edges_csv, "r", encoding="utf-8")
    csv_parser = CsvGraphParser(fn, fe, id_field=ID_FIELD)
    graph = csv_parser.parse()
    csv_parser.cleanup()
    return to_ast(graph.node_by_index(0))


class JoernSourceCode(SourceCode):
    def __init__(self, source, ast, ast_path=None):
        super().__init__(source._author, source._code, source._path)
        self._ast = ast
        self._ast_path = ast_path

    def fetch_ast(self):
        if self._ast is None:
            return self.prefetch_ast()
        return self._ast

    def ast_path(self):
        return self._ast_path

    def set_ast_path(self, path):
        self._ast_path = path

    def prefetch(self):
        super().prefetch()
        self.prefetch_ast()

    def unfetch(self):
        super().unfetch()
        self.unfetch_ast()

    def prefetch_ast(self):
        if self.ast_path() is not None and self._ast is None:
            cached_nodes = NODES_CACHE.format(self.ast_path())
            cached_edges = EDGES_CACHE.format(self.ast_path())
            self._ast = fetch_joern_data_from(cached_nodes, cached_edges)
        return self._ast

    def unfetch_ast(self):
        if self.ast_path() is not None and self._ast is not None:
            self._ast = None
