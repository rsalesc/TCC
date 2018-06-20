import os
import shlex
import subprocess
import tempfile

from . import ParserError, Error
from ..graph.graph import CsvGraphParser
from ..graph.astree import to_ast
from .source import JoernSourceCode


CODENAME = "code"
PARSED_FOLDER = "parsed"
NODES_FILE = "nodes.csv"
EDGES_FILE = "edges.csv"
ID_FIELD = "key"


def run_joern(filepath):
    filepath = os.path.abspath(filepath)
    dirpath = os.path.dirname(filepath)
    args = shlex.split("joern-parse '{}' --dir '{}'".format(dirpath, dirpath))
    returncode = subprocess.call(args)
    if returncode != 0:
        raise ParserError(
            "Parsed exited with non-zero code: {}".format(returncode))


class JoernParser():
    def __init__(self, source):
        self._source = source

    def parse(self):
        with tempfile.TemporaryDirectory() as d:
            filepath = os.path.join(d, CODENAME)
            with open(filepath, "w") as f:
                f.write(self._source.fetch())
            run_joern(filepath)
            parsedpath = os.path.join(d, PARSED_FOLDER)
            if not os.path.isdir(parsedpath):
                raise Error("`parsed` dir was not created under {}".format(d))
            innerpath = os.path.join(parsedpath, CODENAME)
            if not os.path.isdir(innerpath):
                raise Error("inner path was not created: {}".format(innerpath))
            
            with open(os.path.join(innerpath, NODES_FILE)) as fn:
                with open(os.path.join(innerpath, EDGES_FILE)) as fe:
                    csv_parser = CsvGraphParser(fn, fe, id_field=ID_FIELD)
                    root = to_ast(csv_parser.parse().node_by_index(0))
                    return JoernSourceCode(self._source, root)
