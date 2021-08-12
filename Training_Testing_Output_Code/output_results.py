# This script was produced by the Polavieja Lab. Original source code can be found here:
#   https://gitlab.com/polavieja_lab/fishandra

import argparse
from fishandra.sample_with_model import sample_with_model, sample_with_model_separate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', type=str)
    parser.add_argument('--separate', action='store_true',
                        help='an output file for each video')
    args = parser.parse_args()

    if args.separate:
        sample_with_model_separate(args.model_folder)
    else:
        sample_with_model(args.model_folder)


if __name__ == '__main__':
    main()
