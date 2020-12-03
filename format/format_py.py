import sys
from yapf.yapflib.yapf_api import FormatCode

if __name__ == "__main__":
    src_file = sys.argv[1]
    tgt_file = sys.argv[2]

    with open(src_file) as f_src, open(tgt_file, 'w+') as f_tgt:
        unformatted = f_src.read()
        formatted, _ = FormatCode(unformatted, style_config='format/setup.cfg')
        f_tgt.write(formatted)