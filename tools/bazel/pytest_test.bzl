"""Small repository-local pytest rule wrapper."""

load("@mindclade_pypi//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

def pytest_test(name, srcs, args, deps = [], data = [], imports = [], **kwargs):
    """Run pytest over explicit runfile paths using the hermetic Python hub."""
    py_test(
        name = name,
        srcs = srcs + ["//tools/bazel:pytest_main.py"],
        main = "//tools/bazel:pytest_main.py",
        args = args,
        data = data,
        deps = deps + [requirement("pytest")],
        imports = imports,
        **kwargs
    )
