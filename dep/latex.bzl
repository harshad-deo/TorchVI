def latex_dep(xs):
    if type(xs) == "string":
        xs = [xs]
    return ["@texlive_texmf__texmf-dist__tex__latex__%s" % x for x in xs]
