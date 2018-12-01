from scpd.datasets import CodeforcesDisjointDatasetBuilder as Builder
from scpd.datasets import CodeforcesDescriptor as Descriptor
from scpd.datasets import cf_codes_plugin, cf_caide_plugin
from scpd.datasets import CodejamDisjointDatasetBuilder as CodejamBuilder
from scpd.datasets import CodejamEasiestDatasetBuilder as CodejamEasiestBuilder
from scpd.datasets import CodejamDescriptor, gcj_codes_plugin


def newest():
    descriptors = [
        Descriptor("training", 1000, 40, path=".cache/training.pkl"),
        Descriptor("validation", 200, 8, path=".cache/validation.pkl"),
        Descriptor("test", 200, 8, path=".cache/test.pkl")
    ]

    plugins = [
        cf_codes_plugin,
        cf_caide_plugin("/usr/include/clang/3.6/include")
    ]

    builder = Builder(descriptors, download=True, plugins=plugins)
    return builder.extract()


def oldest():
    return preloaded(["submissions_training.dat", "submissions_test.dat"])


def gcj_random():
    descriptors = [
        CodejamDescriptor("training", 512, 12, path=".cache/gcj.training.pkl"),
        CodejamDescriptor("validation", 64, 8,
                          path=".cache/gcj.validation.pkl"),
        CodejamDescriptor("test", 128, 8, path=".cache/gcj.test.pkl")
    ]

    plugins = [
        gcj_codes_plugin,
        cf_caide_plugin("/usr/include/clang/3.6/include")
    ]
    years = [2014]
    lang = "cpp"

    builder = CodejamBuilder(descriptors, years, lang,
                             at_least=8, plugins=plugins)
    return builder.extract()


def gcj_easiest():
    descriptors = [
        CodejamDescriptor("easiest", 250, (8, 1, 2),
                          path=".cache/gcj.easiest.pkl")
    ]

    plugins = [gcj_codes_plugin]
    years = [2014]
    lang = "cpp"

    builder = CodejamEasiestBuilder(descriptors, years, lang, plugins=plugins)
    return builder.extract()


def preloaded(paths, caide=False):
    descriptors = [Descriptor(path, None, None, path=path) for path in paths]

    plugins = [cf_codes_plugin]
    if caide:
        plugins.append(cf_caide_plugin(
            "/usr/include/clang/3.6/include", use_cache=True))

    builder = Builder(descriptors, download=False, plugins=plugins)
    return builder.extract()


if __name__ == "__main__":
    print(len(gcj_easiest()[2]))
