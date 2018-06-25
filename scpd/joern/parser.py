import os
import pickle
import shlex
import subprocess
import tempfile
import traceback
from tqdm import tqdm

from . import ParserError, Error
from .. import utils
from ..config import JOERN_PARSE_PATH
from ..graph.graph import CsvGraphParser
from ..graph.astree import to_ast
from .source import (JoernSourceCode, fetch_joern_data_from,
                     check_joern_data_from)

DIRNAME = "inner"
CODENAME = "code.cpp"
PARSED_FOLDER = "parsed"
NODES_FILE = "nodes.csv"
EDGES_FILE = "edges.csv"
NODES_CACHE = "{}.n.ast"
EDGES_CACHE = "{}.e.ast"
ID_FIELD = "key"


def run_joern(filepath, outdir):
    filepath = os.path.abspath(filepath)
    outdir = os.path.abspath(outdir)
    args = [
        "{}".format(JOERN_PARSE_PATH),
        "{}".format(os.path.dirname(filepath)),
        "--outdir {}".format(outdir),
    ]
    returncode = subprocess.call(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if returncode != 0:
        raise ParserError(
            "Parser exited with non-zero code: {}".format(returncode))


def extract_joern_code(code, cache_along=True, force=False):
    return JoernParser(code, cache_along=cache_along).parse(force=force)


class JoernParser():
    def __init__(self, source, cache_along=True):
        self._source = source
        self._cache = cache_along and self._source.path() is not None

    def parse(self, force=False):
        if self._source.path() is not None and not force:
            cached_nodes = NODES_CACHE.format(self._source.path())
            cached_edges = EDGES_CACHE.format(self._source.path())
            if not os.path.isfile(cached_nodes) or not os.path.isfile(
                    cached_edges):
                force = True

        if force:
            with tempfile.TemporaryDirectory() as d:
                innerdir = os.path.join(d, DIRNAME)
                os.makedirs(innerdir)
                filepath = os.path.join(innerdir, CODENAME)
                with open(filepath, "w") as f:
                    f.write(self._source.fetch())
                outdir = os.path.join(d, PARSED_FOLDER)
                run_joern(filepath, outdir)
                if not os.path.isdir(outdir):
                    raise Error(
                        "`parsed` dir was not created under {}".format(d))
                parsedpath = outdir + filepath
                if not os.path.isdir(parsedpath):
                    raise Error("inner parsed path was not created: {}".format(
                        parsedpath))

                nodes_file = os.path.join(parsedpath, NODES_FILE)
                edges_file = os.path.join(parsedpath, EDGES_FILE)
                if not self._cache:
                    return JoernSourceCode(
                        self._source,
                        ast=fetch_joern_data_from(nodes_file, edges_file))
                else:
                    cached_nodes = NODES_CACHE.format(self._source.path())
                    cached_edges = EDGES_CACHE.format(self._source.path())

                    utils.copies(nodes_file, cached_nodes)
                    utils.copies(edges_file, cached_edges)

        # we should check data anyways to make sure it can be parsed
        cached_nodes = NODES_CACHE.format(self._source.path())
        cached_edges = EDGES_CACHE.format(self._source.path())
        check_joern_data_from(cached_nodes, cached_edges)

        return JoernSourceCode(
            self._source, ast=None, ast_path=self._source.path())


class BatchJoernParser():
    def __init__(self, pool, codes, monitor=True, cache_along=True):
        self._pool = pool
        self._codes = codes
        self._monitor = monitor
        self._cache = cache_along

    def parse(self, force=False):
        futures = []
        for code in self._codes:
            future = self._pool.submit(
                extract_joern_code, code, cache_along=self._cache, force=force)
            futures.append(future)

        sources = []
        enumerated = enumerate(futures)
        if self._monitor:
            enumerated = tqdm(
                enumerate(futures), total=len(futures), desc="Joern Parsing")
        for i, future in enumerated:
            if self._monitor:
                enumerated.set_postfix(skipped="{}".format(i - len(sources)))
            # TODO: better exception handling
            try:
                sources.append(future.result(timeout=5))
            except (Error, ParserError):
                pass
            except:
                raise
        if self._monitor:
            print("Skipped {} entries...".format(len(futures) - len(sources)))
        return sources
