import os
from argparse import ArgumentParser
from math import ceil
from concurrent.futures import (
    ThreadPoolExecutor
)
import json
import nlpaug.augmenter.word as naw
from tqdm import tqdm


class AugmentOptions():
    synonym_aug: naw.SynonymAug = None
    max_threads: int = 1
    max_passes: int = 1


def append_jsonl_to_file(
    jsonl: dict,
    file: str
):
    try:
        with open(file, "a") as file:
            file.write(json.dumps(jsonl) + "\n")
    except Exception as e:
        print(f'''
            Exception occurred while appending to {output_file}:
            {e}
        ''')


def augment_jsonl_from_string(
    options: AugmentOptions,
    data: str
) -> dict:
    try:
        jsonl = json.loads(data)
    except Exception as e:
        print(f"Exception occurred loading line as json object: {e}")
    else:
        if "instruction" not in jsonl:
            raise Exception("jsonl data has no 'instruction'")
        if "output" not in jsonl:
            raise Exception("jsonl data has no 'output'")
        try:
            augmented = options.synonym_aug.augment(
                data=[
                    jsonl["instruction"],
                    jsonl["output"]
                ]
            )

            # Replace the jsonl keys with the augmented versions
            jsonl["instruction"] = augmented[0]
            jsonl["output"] = augmented[1]
        except Exception as e:
            print(f"Exception occurred while augmenting jsonl object: {e}")
        else:
            return jsonl


def augment_jsonl_file(
    options: AugmentOptions,
    file: str,
    destination: str
):
    # Make max threads match max passes if max passes is less than
    # max threads (Don't use more threads than needed basically)
    max_threads = min(options.max_threads, options.max_passes)
    max_passes = ceil(options.max_passes / max_threads)

    print(f"Using {max_threads} threads.")

    with open(file, "r") as file:
        line_number = 0
        for line in tqdm(file):
            line_number += 1
            print(f"Processing line {line_number}, {line}", flush=True)

            for num_pass in range(max_passes):
                if max_passes > 1:
                    print(
                        f"pass {num_pass+1}, line {line_number}",
                        flush=True
                    )

                with ThreadPoolExecutor(
                    max_workers=max_threads
                ) as executor:
                    for future in tqdm([
                        executor.submit(
                            augment_jsonl_from_string,
                            options,
                            line
                        ) for _ in range(max_threads)
                    ]):
                        try:
                            augmented_jsonl = future.result()
                        except Exception as e:
                            print(f'''
                                Exception occurred when augmenting line: {line}
                                {e}
                            ''')
                        else:
                            append_jsonl_to_file(augmented_jsonl, destination)


if __name__ == "__main__":
    parser = ArgumentParser()
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

    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help='''
            If the output file exists,
            overwrite it instead of
            throwing an exception.
        '''
    )

    args = parser.parse_args()
    file = args.file
    file_name = os.path.basename(file)
    output_dir = args.output

    if not os.path.exists(file):
        raise Exception("--file specified is an invalid file.")
    if not os.path.exists(output_dir):
        raise Exception("--output specified is a path that doesn't exist.")
    if not os.path.isdir(output_dir):
        raise Exception("--output specified is not a directory.")

    output_file = os.path.join(output_dir, file_name)
    if os.path.exists(output_file):
        if args.overwrite:
            print(f'''
                Overwriting existing {output_file}
                since --overwrite was used.
            ''')

            with open(output_file, "w"):
                pass
        else:
            raise Exception(f'''
                {output_file} already exists.
                Please move or rename it and try running again.
            ''')

    # Don't allow user input to exceed processor count
    max_threads = min(args.max_threads, max(1, cpu_count))

    aug_options = AugmentOptions()
    aug_options.synonym_aug = naw.SynonymAug(aug_p=0.2)
    aug_options.max_threads = max_threads
    aug_options.max_passes = args.max_passes

    augment_jsonl_file(aug_options, file, output_file)
