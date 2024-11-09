import os
from shutil import copy as shutil_copy
from argparse import ArgumentParser
from math import ceil
from concurrent.futures import (
    ThreadPoolExecutor  # for processing multiple lines of a file (all threads)
)
import asyncio          # for augmenting properties of a line and IO (1 thread)
import json
import nlpaug.augmenter.word as naw
from tqdm import tqdm


class AugmentOptions():
    synonym_aug: naw.SynonymAug = None
    keys: list[str] = []
    max_threads: int = 1
    max_passes: int = 1


# TODO: make asynchronous
# we should not block thread
# just to append to a file..
def append_jsonl_to_file(
    jsonl: dict,
    file: str
):
    try:
        with open(file, "a") as open_file:
            open_file.write(json.dumps(jsonl) + "\n")
    except Exception as e:
        print(f'''
            Exception occurred while appending to {output_file}:
            {e}
        ''')


# Augment a string that has jsonl structure
def augment_jsonl_from_string(
    options: AugmentOptions,
    data: str
) -> dict:
    try:
        async def augment(key, value) -> dict:
            try:
                augmented = options.synonym_aug.augment(value)
                return {key: augmented[0]}
            except Exception as e:
                print(f'''
                    Exception occurred when augmenting key: {key}

                    {e}
                ''')
                return {}

        async def make_code_async() -> dict:
            jsonl = json.loads(data)
            augmented_jsonl: dict = {}
            augment_tasks: list[asyncio.Task] = []

            for key, value in jsonl.items():
                if key in options.keys:
                    augment_tasks.append(
                        asyncio.create_task(augment(key, value))
                    )
                else:
                    # we don't want to augment this,
                    # so just add it as is
                    augmented_jsonl[key] = value

            # Merge augmented key values into one dict, return it
            for result in await asyncio.gather(*augment_tasks):
                augmented_jsonl.update(result)

            return augmented_jsonl

        return asyncio.run(make_code_async())
    except Exception as e:
        print(f'''
            Error augmenting jsonl string: {data}

            {e}
        ''')
    return {}


def augment_jsonl_file(
    options: AugmentOptions,
    file: str,
    destination: str
):

    # Make max threads match max passes if max passes is less than
    # max threads (Don't use more threads than needed basically)
    max_threads = min(options.max_threads, options.max_passes)
    max_passes_per_thread = ceil(options.max_passes / max_threads)

    print(f"Using {max_threads} threads.")

    with open(file, "r") as open_file:
        line_number = 0
        for line in tqdm(open_file):
            append_count = 0
            line_number += 1
            print(f"Processing line {line_number}, {line}", flush=True)

            for num_pass in range(max_passes_per_thread):
                if max_passes_per_thread > 1:
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
                            if (append_count >= options.max_passes):
                                break

                            augmented_jsonl = future.result()
                            append_count += 1
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
        required=True,
        type=str,
        help="Jsonl file to augment"
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Directory to store augmented file"
    )

    parser.add_argument(
        "-nsc",
        "--nlpaug_synonym_config",
        required=True,
        type=str,
        help='''
            File that contains the settings for
            nlpaug's SynonymAug

            More info here:
            https://nlpaug.readthedocs.io/en/latest/
            augmenter/word/synonym.html?highlight=synonymaug
            #nlpaug.augmenter.word.synonym.SynonymAug
        '''
    )

    parser.add_argument(
        "-k",
        "--key",
        required=True,
        type=str,
        action="append",
        help='''
            Specify a key that we are allowed to
            augment from the jsonl file.
        '''
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
    nlpaug_synonym_file = args.nlpaug_synonym_config
    file = args.file
    file_name = os.path.basename(file)
    output_dir = args.output

    if not os.path.exists(nlpaug_synonym_file):
        raise FileNotFoundError(
            "--nlpaug_synonym_config points to a nonexistent file."
        )
    if os.path.isdir(nlpaug_synonym_file):
        raise IsADirectoryError(
            "--nlpaug_synonym_config specified is not pointing to a file."
        )

    if not os.path.exists(file):
        raise FileNotFoundError("--file specified is an invalid file.")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(
            "--output specified is a path that doesn't exist."
        )
    if not os.path.isdir(output_dir):
        raise NotADirectoryError("--output specified is not a directory.")

    output_file = os.path.join(output_dir, file_name)
    if os.path.exists(output_file):
        if args.overwrite:
            print(f'''
                Overwriting existing {output_file}
                since --overwrite was used.
            ''')
        else:
            raise FileExistsError(f'''
                {output_file} already exists.
                Please move or rename it and try running again.
            ''')

    print(f"Copying existing dataset {file} to {output_file}..")
    shutil_copy(file, output_file)
    print("Finished copy")

    if isinstance(args.max_threads, int) and isinstance(cpu_count, int):
        # Don't allow user input to exceed processor count
        max_threads = min(args.max_threads, max(1, cpu_count))
    else:
        raise ValueError('''
            cpu_count and max_threads must be a valid integer
        ''')

    aug_options = AugmentOptions()
    aug_options.keys = args.key
    aug_options.max_threads = max_threads
    aug_options.max_passes = args.max_passes
    with open(nlpaug_synonym_file, "r") as config:
        aug_options.synonym_aug = naw.SynonymAug(
            **json.loads(config.read())
        )

    augment_jsonl_file(aug_options, file, output_file)
