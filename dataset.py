from scpd.datasets import CodeforcesDisjointDatasetBuilder as Builder
from scpd.datasets import CodeforcesDescriptor as Descriptor
from scpd.datasets import cf_codes_plugin, cf_joern_plugin, cf_caide_plugin


def newest():
    descriptors = [
        Descriptor("training", 1000, 40, path=".cache/training.pkl"),
        Descriptor("validation", 200, 8, path=".cache/validation.pkl"),
        Descriptor("test", 200, 8, path=".cache/test.pkl")
    ]

    plugins = [cf_codes_plugin]

    builder = Builder(descriptors, download=True, plugins=plugins)
    builder.extract()


def oldest():
    return preloaded(["submissions_training.dat", "submissions_test.dat"])


def preloaded(paths):
    descriptors = [Descriptor(path, None, None, path=path) for path in paths]

    plugins = [
        cf_codes_plugin,
        cf_caide_plugin("/usr/include/clang/3.6/include")
    ]

    builder = Builder(descriptors, download=False, plugins=plugins)
    return builder.extract()


if __name__ == "__main__":
    oldest()
