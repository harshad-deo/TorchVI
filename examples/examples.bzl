load("@py_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

common_deps = [
    requirement("matplotlib"),
    requirement("pyparsing"),
    requirement("cycler"),
    requirement("python-dateutil"),
    requirement("kiwisolver"),
    requirement("Pillow"),
    requirement("torch"),
    requirement("numpy"),
    requirement("tqdm"),
    "//torchvi",
    "//torchvi/vtensor",
    "//torchvi/vdistributions",
    "//utils:fix_seed",
]

pandas_deps = [
    requirement("pandas"),
    requirement("pytz"),
    requirement("numexpr"),
    requirement("bottleneck"),
]

def simple_example(name, with_pandas = False, data = []):
    file_name_py = name + ".py"
    all_deps = common_deps
    if with_pandas:
        all_deps = all_deps + pandas_deps
    py_binary(
        name = name,
        srcs = [file_name_py],
        deps = all_deps,
        main = file_name_py,
        data = data,
    )
