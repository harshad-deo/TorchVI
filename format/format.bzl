def _replace_formatted(ctx, manifest, files):
    out = ctx.actions.declare_file(ctx.label.name)

    # this makes it easier to add variables
    file_lines = [
        """#!/bin/bash -e
WORKSPACE_ROOT="${1:-$BUILD_WORKSPACE_DIRECTORY}" """,
        """RUNPATH="${TEST_SRCDIR-$0.runfiles}"/""" + ctx.workspace_name,
        """RUNPATH=(${RUNPATH//bin/ })
RUNPATH="${RUNPATH[0]}"bin
echo $WORKSPACE_ROOT
echo $RUNPATH
while read original formatted; do
    if [[ ! -z "$original" ]] && [[ ! -z "$formatted" ]]; then
        if ! cmp -s "$WORKSPACE_ROOT/$original" "$RUNPATH/$formatted"; then
            echo "Formatting $original"
            cp "$RUNPATH/$formatted" "$WORKSPACE_ROOT/$original"
        fi
    fi
done < "$RUNPATH"/""" + manifest.short_path,
    ]

    file_content = "\n".join(file_lines)

    ctx.actions.write(
        output = out,
        content = file_content,
    )
    files.append(manifest)
    return [DefaultInfo(files = depset(files), executable = out)]

def _build_format_py(ctx):
    files = []
    manifest_content = []

    for src in ctx.files.srcs:
        if src.is_source:
            file = ctx.actions.declare_file("{}.format.output".format(src.short_path))
            files.append(file)
            ctx.actions.run(
                arguments = [src.path, file.path],
                executable = ctx.executable._fmt,
                outputs = [file],
                inputs = [src, ctx.file._style],
            )
            manifest_content.append("{} {}".format(src.short_path, file.short_path))

    manifest = ctx.actions.declare_file("format/{}/manifest.txt".format(ctx.label.name))
    ctx.actions.write(manifest, "\n".join(manifest_content) + "\n")

    return manifest, files

def _format_py_impl(ctx):
    manifest, files = _build_format_py(ctx)
    return _replace_formatted(ctx, manifest, files)

format_py = rule(
    implementation = _format_py_impl,
    executable = True,
    attrs = {
        "srcs": attr.label_list(
            allow_files = [".py"],
            mandatory = True,
        ),
        "_fmt": attr.label(
            cfg = "host",
            default = "//format:format_py",
            executable = True,
        ),
        "_style": attr.label(
            allow_single_file = True,
            default = ":setup.cfg",
        ),
    },
)
