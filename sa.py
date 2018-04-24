# from pathlib import PosixPath
# import sys
# path = PosixPath(__file__).absolute().parents[1].as_posix()
# sys.path.extend([path])
#
# from Ernn.tokenizers import CoreNLPTokenizer
# tok = CoreNLPTokenizer()
# print(tok.tokenize('hello world').words())
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('outdir', type=str,default='predictions.json',
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))
args = parser.parse_args()
outfile = args.outdir
print('Writing results to %s' % outfile)
result = {'classification': 'mmp'}
with open(outfile, 'w') as f:
    json.dump(result, f)