load("@py_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

def simple_example(name):
    file_name_py = name + ".py"
    py_binary(
        name = name,
        srcs = [file_name_py],
        deps = [
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
            "//utils:fix_seed",
        ],
        main = file_name_py,
    )
