load("@bazel_latex//:latex.bzl", "latex_document")
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
            "//utils:fix_seed",
        ],
        main = file_name_py,
    )

    name_tex = name + "_doc"
    file_name_tex = file_name + ".tex"
    latex_document(
        name = name_tex,
        srcs = [
            "//dep:mathtools",
            "//dep:setspace",
            "@bazel_latex//packages:amsmath",
            "@bazel_latex//packages:amssymb",
            "@bazel_latex//packages:calc",
            "@bazel_latex//packages:geometry",
            "@bazel_latex//packages:hyperref",
        ],
        main = file_name_tex,
    )
