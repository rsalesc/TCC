import tempfile
import os
import subprocess
import shlex
from tqdm import tqdm

from .. import utils

CMD_PATH = "optimizer/build/cmd/cmd"
FLAGS = "-std=c++14 -I{include} -Wno-everything -fparse-all-comments -- -l {lines} -d {dir} -o {output}"
INPUT_NAME = "input.cpp"
OUTPUT_NAME = "output.cpp"


class OptimizerError(Exception):
    pass


class SkippedError(Exception):
    pass


class CodeOptimizer:
    def __init__(self, clang_includes, verbose=False, lines=1, **kwargs):
        self._includes = clang_includes
        self._verbose = verbose
        self._lines = 1

    def run(self, code):
        with tempfile.TemporaryDirectory() as d:
            input_path = os.path.join(d, INPUT_NAME)
            output_path = os.path.join(d, OUTPUT_NAME)
            with open(input_path, "w") as f:
                f.write(code)
            args = shlex.split("{} {} {}".format(
                CMD_PATH,
                FLAGS.format(
                    include=self._includes,
                    dir=d,
                    output=output_path,
                    lines=self._lines), input_path))
            returncode = subprocess.call(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if returncode != 0:
                raise OptimizerError(
                    "Caide exited with non-zero code: {}".format(returncode))
            return open(output_path, "r", encoding="utf-8").read()


class BatchSourceOptimizer():
    def __init__(self, pool, optimizer, monitor=True, cache_along=True,
                 use_cache=False, **kwargs):
        self._pool = pool
        self._optimizer = optimizer
        self._monitor = monitor
        self._cache = cache_along
        self._use_cache = use_cache

    def run(self, sources, force, load=True):
        futures = []

        def extract(source, cache_along):
            if source.path() is not None:
                path = os.path.abspath(source.path())
                caide_path = "{}.caide".format(path)
                if os.path.isfile(caide_path) and not force:
                    source._path = caide_path
                    return source
            if self._use_cache:
                return source
            code = source.fetch()
            caide_code = self._optimizer.run(code)
            if source.path() is not None and cache_along:
                path = os.path.abspath(source.path())
                caide_path = "{}.caide".format(path)
                with utils.opens(caide_path, "w") as f:
                    f.write(caide_code)
                source._path = caide_path
            else:
                source._code = caide_code
            return source

        for source in sources:
            future = self._pool.submit(
                extract, source, cache_along=self._cache)
            futures.append(future)

        result = []
        enumerated = enumerate(futures)
        skipped = 0
        if self._monitor:
            enumerated = tqdm(
                enumerate(futures),
                total=len(futures),
                desc="Caide Optimization")
        for i, future in enumerated:
            if self._monitor:
                enumerated.set_postfix(skipped=skipped)
            # TODO: better exception handling
            try:
                sources[i] = future.result(timeout=5)
            except (OptimizerError, ):
                skipped += 1
                pass
            except:
                raise
        if self._monitor:
            print("Skipped {} entries...".format(skipped))
        return result


if __name__ == "__main__":
    optimizer = CodeOptimizer("/usr/include/clang/3.6/include")
    print(optimizer.run("// maoe\n int main() {}"))
