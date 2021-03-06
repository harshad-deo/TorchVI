workspace(name = "torchvi")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//rules:http_archive_ext.bzl", "http_archive_ext")

http_archive(
    name = "rules_python",
    sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "d1ffd055969c8f8d431e2d439813e42326961d0942bdf734d2c95dc30c369566",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.24.5/rules_go-v0.24.5.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.24.5/rules_go-v0.24.5.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains()

http_archive(
    name = "bazel_gazelle",
    sha256 = "b85f48fa105c4403326e9525ad2b2cc437babaa6e15a3fc0b1dbab0ab064bc7c",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.22.2/bazel-gazelle-v0.22.2.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.22.2/bazel-gazelle-v0.22.2.tar.gz",
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

http_archive(
    name = "com_google_protobuf",
    sha256 = "71030a04aedf9f612d2991c1c552317038c3c5a2b578ac4745267a45e7037c29",
    strip_prefix = "protobuf-3.12.3",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.12.3.tar.gz"],
)

load(
    "@com_google_protobuf//:protobuf_deps.bzl",
    "protobuf_deps",
)

protobuf_deps()

http_archive(
    name = "com_github_bazelbuild_buildtools",
    strip_prefix = "buildtools-master",
    url = "https://github.com/bazelbuild/buildtools/archive/master.zip",
)

http_archive(
    name = "bazel_latex",
    sha256 = "f81604ec9318364c05a702798c5507c6e5257e851d58237d5f171eeca4d6e2db",
    strip_prefix = "bazel-latex-1.0",
    url = "https://github.com/ProdriveTechnologies/bazel-latex/archive/v1.0.tar.gz",
)

load("@bazel_latex//:repositories.bzl", "latex_repositories")

latex_repositories()

http_archive(
    name = "rules_pkg",
    sha256 = "aeca78988341a2ee1ba097641056d168320ecc51372ef7ff8e64b139516a4937",
    urls = [
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.6-1/rules_pkg-0.2.6.tar.gz",
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.2.6/rules_pkg-0.2.6.tar.gz",
    ],
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

http_archive_ext(
    name = "python39_static",
    build_file_content = """
exports_files(["python_bin"])
filegroup(
    name = "files",
    srcs = glob(["build/**"], exclude = ["**/* *"]),
    visibility = ["//visibility:public"],
)
""",
    patch_cmds = [
        "mkdir $(pwd)/build",
        "./configure --prefix=$(pwd)/build --enable-optimizations --disable-shared --enable-option-checking=fatal --with-lto --without-ensurepip",
        "make -j6",
        "make install",
        "ln -s build/bin/python3 python_bin",
    ],
    sha256 = "29cb91ba038346da0bd9ab84a0a55a845d872c341a4da6879f462e94c741f117",
    strip_prefix = "Python-3.9.1",
    urls = [
        "https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz",
    ],
)

load("@rules_python//python:pip.bzl", "pip_install")

pip_install(
    name = "py_deps",
    python_interpreter_target = "@python39_static//:python_bin",
    requirements = "//:requirements.txt",
)

http_file(
    name = "poisson_sim",
    downloaded_file_path = "poisson_sim.csv",
    sha256 = "567f8cbb08132a92bd75e36bf9a13afb336f11783bd24698959809a82eacf33b",
    urls = ["https://stats.idre.ucla.edu/stat/data/poisson_sim.csv"],
)

register_toolchains("//:custom_py_toolchain")
