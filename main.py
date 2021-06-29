import sys
sys.path.append('agent')
sys.path.append('datasets')
sys.path.append('models')
sys.path.append('utils')
import argparse
from agent import train

parser = argparse.ArgumentParser(description='YOLO v1')
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    if args.train:
        train()
    # else if args.eval:
    #     eval()

