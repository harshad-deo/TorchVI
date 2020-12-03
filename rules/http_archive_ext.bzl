load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "read_netrc",
    "update_attrs",
    "use_netrc",
    "workspace_and_buildfile",
)

def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def _use_native_patch(patch_args):
    """If patch_args only contains -p<NUM> options, we can use the native patch implementation."""
    for arg in patch_args:
        if not arg.startswith("-p"):
            return False
    return True

def patch(ctx, patches = None, patch_cmds = None, patch_cmds_win = None, patch_tool = None, patch_args = None):
    """Implementation of patching an already extracted repository.

    This rule is inteded to be used in the implementation function of
    a repository rule. Ifthe parameters `patches`, `patch_tool`,
    `patch_args`, `patch_cmds` and `patch_cmds_win` are not specified
    then they are taken from `ctx.attr`.

    Args:
      ctx: The repository context of the repository rule calling this utility
        function.
      patches: The patch files to apply. List of strings, Labels, or paths.
      patch_cmds: Bash commands to run for patching, passed one at a
        time to bash -c. List of strings
      patch_cmds_win: Powershell commands to run for patching, passed
        one at a time to powershell /c. List of strings. If the
        boolean value of this parameter is false, patch_cmds will be
        used and this parameter will be ignored.
      patch_tool: Path of the patch tool to execute for applying
        patches. String.
      patch_args: Arguments to pass to the patch tool. List of strings.

    """
    bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
    powershell_exe = ctx.os.environ["BAZEL_POWERSHELL"] if "BAZEL_POWERSHELL" in ctx.os.environ else "powershell.exe"

    if patches == None and hasattr(ctx.attr, "patches"):
        patches = ctx.attr.patches
    if patches == None:
        patches = []

    if patch_cmds == None and hasattr(ctx.attr, "patch_cmds"):
        patch_cmds = ctx.attr.patch_cmds
    if patch_cmds == None:
        patch_cmds = []

    if patch_cmds_win == None and hasattr(ctx.attr, "patch_cmds_win"):
        patch_cmds_win = ctx.attr.patch_cmds_win
    if patch_cmds_win == None:
        patch_cmds_win = []

    if patch_tool == None and hasattr(ctx.attr, "patch_tool"):
        patch_tool = ctx.attr.patch_tool
    if not patch_tool:
        patch_tool = "patch"
        native_patch = True
    else:
        native_patch = False

    if patch_args == None and hasattr(ctx.attr, "patch_args"):
        patch_args = ctx.attr.patch_args
    if patch_args == None:
        patch_args = []

    if len(patches) > 0 or len(patch_cmds) > 0:
        ctx.report_progress("Patching repository")

    if native_patch and _use_native_patch(patch_args):
        if patch_args:
            strip = int(patch_args[-1][2:])
        else:
            strip = 0
        for patchfile in patches:
            ctx.patch(patchfile, strip)
    else:
        for patchfile in patches:
            command = "{patchtool} {patch_args} < {patchfile}".format(
                patchtool = patch_tool,
                patchfile = ctx.path(patchfile),
                patch_args = " ".join([
                    "'%s'" % arg
                    for arg in patch_args
                ]),
            )
            st = ctx.execute([bash_exe, "-c", command])
            if st.return_code:
                fail("Error applying patch %s:\n%s%s" %
                     (str(patchfile), st.stderr, st.stdout))

    if _is_windows(ctx) and patch_cmds_win:
        for cmd in patch_cmds_win:
            st = ctx.execute([powershell_exe, "/c", cmd])
            if st.return_code:
                fail("Error applying patch command %s:\n%s%s" %
                     (cmd, st.stdout, st.stderr))
    else:
        for cmd in patch_cmds:
            st = ctx.execute([bash_exe, "-c", cmd], timeout = 7200)
            if st.return_code:
                fail("Error applying patch command %s:\n%s%s" %
                     (cmd, st.stdout, st.stderr))

def _get_auth(ctx, urls):
    """Given the list of URLs obtain the correct auth dict."""
    if ctx.attr.netrc:
        netrc = read_netrc(ctx, ctx.attr.netrc)
        return use_netrc(netrc, urls, ctx.attr.auth_patterns)

    if "HOME" in ctx.os.environ and not ctx.os.name.startswith("windows"):
        netrcfile = "%s/.netrc" % (ctx.os.environ["HOME"])
        if ctx.execute(["test", "-f", netrcfile]).return_code == 0:
            netrc = read_netrc(ctx, netrcfile)
            return use_netrc(netrc, urls, ctx.attr.auth_patterns)

    if "USERPROFILE" in ctx.os.environ and ctx.os.name.startswith("windows"):
        netrcfile = "%s/.netrc" % (ctx.os.environ["USERPROFILE"])
        if ctx.path(netrcfile).exists:
            netrc = read_netrc(ctx, netrcfile)
            return use_netrc(netrc, urls, ctx.attr.auth_patterns)

    return {}

def _http_archive_ext_impl(ctx):
    """Implementation of the http_archive rule."""
    if not ctx.attr.url and not ctx.attr.urls:
        fail("At least one of url and urls must be provided")
    if ctx.attr.build_file and ctx.attr.build_file_content:
        fail("Only one of build_file and build_file_content can be provided.")

    all_urls = []
    if ctx.attr.urls:
        all_urls = ctx.attr.urls
    if ctx.attr.url:
        all_urls = [ctx.attr.url] + all_urls

    auth = _get_auth(ctx, all_urls)

    download_info = ctx.download_and_extract(
        all_urls,
        "",
        ctx.attr.sha256,
        ctx.attr.type,
        ctx.attr.strip_prefix,
        canonical_id = ctx.attr.canonical_id,
        auth = auth,
    )
    workspace_and_buildfile(ctx)
    patch(ctx)

    return update_attrs(ctx.attr, _http_archive_attrs.keys(), {"sha256": download_info.sha256})

_http_archive_attrs = {
    "url": attr.string(),
    "urls": attr.string_list(),
    "sha256": attr.string(),
    "netrc": attr.string(),
    "auth_patterns": attr.string_dict(),
    "canonical_id": attr.string(),
    "strip_prefix": attr.string(),
    "type": attr.string(),
    "patches": attr.label_list(
        default = [],
    ),
    "patch_tool": attr.string(
        default = "",
    ),
    "patch_args": attr.string_list(
        default = ["-p0"],
    ),
    "patch_cmds": attr.string_list(
        default = [],
    ),
    "patch_cmds_win": attr.string_list(
        default = [],
    ),
    "build_file": attr.label(
        allow_single_file = True,
    ),
    "build_file_content": attr.string(),
    "workspace_file": attr.label(),
    "workspace_file_content": attr.string(),
}

http_archive_ext = repository_rule(
    implementation = _http_archive_ext_impl,
    attrs = _http_archive_attrs,
)
