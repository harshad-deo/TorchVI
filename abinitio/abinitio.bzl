load("@py_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

def simple_example(file_name):
    splits = file_name.split("_")
    splits = splits[1:]
    name = "_".join(splits)
    file_name_py = file_name + ".py"

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
            "//utils:fix_seed"
        ],
        main = file_name_py,
    )
