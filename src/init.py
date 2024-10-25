import argparse
import os
import json
import nlpaug.augmenter.word as naw

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--file",
    type=str,
    help="Jsonl file to augment"
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Directory to store augmented file"
)

parser.add_argument(
    "--maxpasses",
    type=int,
    default=1,
    help="How many times to rephrase every line in the file."
)

args = parser.parse_args()
file = args.file
output = args.output

if not os.path.exists(file):
    raise Exception("--file specified is an invalid file.")
if not os.path.exists(output) or not os.path.isdir(output):
    raise Exception("--output specified is not a directory.")

print("Ready to start reading jsonl.")

# 1. For each line in the jsonl file, create a thread that will rephrase it, append to output file
# 2. Run code in a loop for max_passes value, default 1

augmenter = naw.SynonymAug(aug_p=0.1)
#augmented_data = augmenter.augment(example)
