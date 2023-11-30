import json
from openai import OpenAI
from lib.io import save_to_json
import setting


client = OpenAI()


def main():
    file_ids = json.load(open(setting.file_id_filename, "r"))
    print(file_ids)

    start_training(
        training_file_id=file_ids["training_file_id"],
        validation_file_id=file_ids["validation_file_id"],
        suffix_name=setting.model_suffix,
    )


def start_training(training_file_id, validation_file_id, suffix_name):
    # Create a Fine Tuning Job
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=setting.base_model_id,
        suffix=suffix_name,
    )

    print(response)
    job_id = response.id
    save_to_json(response, setting.job_id_filename)
    return job_id


if __name__ == "__main__":
    main()
