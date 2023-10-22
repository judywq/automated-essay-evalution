import json
import openai
from lib.utils import convert_essay, format_check
from lib.stats import print_stats
import setting


def main():
    file_ids = json.load(open(setting.uploaded_file_id_filename, "r"))
    print(file_ids)
    start_training(
        training_file_id=file_ids["training_file_id"],
        validation_file_id=file_ids["validation_file_id"],
        suffix_name=setting.model_suffix,
    )


def start_training(training_file_id, validation_file_id, suffix_name):
    # Create a Fine Tuning Job
    response = openai.FineTuningJob.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="gpt-3.5-turbo",
        suffix=suffix_name,
    )

    print(response)
    job_id = response["id"]
    json.dump(response, open(setting.job_id_filename, "w"))
    return job_id


if __name__ == "__main__":
    main()
