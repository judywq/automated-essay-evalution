from icecream import ic
from lib.essay import Essay
from lib.stats import print_stats
from lib.utils import convert_essay, format_check, divide_chunks
from lib.io import save_to_jsonl
import setting


def prepare_datasets():
    _prepare(
        input_file=setting.index_train_filename,
        chunk_size=setting.num_of_essays_per_prompt,
        system_message=setting.system_message,
        dataset_fn=setting.dataset_train_filename,
    )
    _prepare(
        input_file=setting.index_val_filename,
        chunk_size=setting.num_of_essays_per_prompt,
        system_message=setting.system_message,
        dataset_fn=setting.dataset_val_filename,
    )
    # No need to generate test dataset
    # prepare(
    #     input_file=setting.index_test_filename,
    #     chunk_size=setting.num_of_essays_per_prompt,
    #     system_message=setting.system_message,
    #     dataset_fn=setting.dataset_test_filename,
    # )


def _prepare(input_file, chunk_size, system_message, dataset_fn):
    essay_list = Essay.load_essays(input_file)
    ic(len(essay_list))
    dataset = []
    for chunk in divide_chunks(essay_list, chunk_size):
        record = convert_essay(chunk, system_message)
        dataset.append(record)
    save_to_jsonl(dataset, dataset_fn)

    # Initial dataset stats
    print("Num examples:", len(dataset))
    format_errors = format_check(dataset)
    if format_errors:
        print(f"Found errors: {format_errors}")
        return

    print_stats(dataset)



if __name__ == "__main__":
    prepare_datasets()
