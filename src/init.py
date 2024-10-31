import os
import argparse
from concurrent.futures import (
    ThreadPoolExecutor
)
import nlpaug.augmenter.word as naw
# import json


class AugmentOptions():
    synonym_aug: naw.SynonymAug = None
    max_threads: int = 1
    max_passes: int = 1


def augment_line(
    options: AugmentOptions,
    line: str,
    destination: str
):
    print(f"Processing {line}", flush=True)
    print(options.synonym_aug.augment(line))


def augment_file(
    options: AugmentOptions,
    file: str,
    destination: str
):
    # Only use the amount of threads we will actually need
    max_threads = min(options.max_threads, options.max_passes)

    print(f"Using {max_threads} threads.")

    with open(file, "r") as file:
        for line in file:
            with ThreadPoolExecutor(
                max_workers=max_threads
            ) as executor:
                for future in [
                    executor.submit(
                        augment_line,
                        options,
                        line,
                        destination
                    ) for _ in range(max_threads)
                ]:
                    try:
                        future.result()
                    except Exception as e:
                        print(f'''
                            Exception occurred when augmenting line: {line}
                            {e}
                        ''')


if __name__ == "__main__":
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
        "--max_passes",
        type=int,
        default=1,
        help="How many times to rephrase every line in the file."
    )

    cpu_count = os.cpu_count()
    parser.add_argument(
        "--max_threads",
        type=int,
        default=cpu_count,
        help="How many CPU threads to make use of."
    )

    args = parser.parse_args()
    file = args.file
    file_name = os.path.basename(file)
    output_dir = args.output

    if not os.path.exists(file):
        raise Exception("--file specified is an invalid file.")
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        raise Exception("--output specified is not a directory.")

    output_file = os.path.join(output_dir, file_name)

    # Don't allow user input to exceed processor count
    max_threads = min(args.max_threads, max(1, cpu_count))

    aug_options = AugmentOptions()
    aug_options.synonym_aug = naw.SynonymAug(aug_p=0.1)
    aug_options.max_threads = max_threads
    aug_options.max_passes = args.max_passes

    augment_file(
        aug_options,
        file,
        output_file
    )
