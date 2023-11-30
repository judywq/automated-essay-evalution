import json
from openai import OpenAI
from lib.io import save_to_json
import setting

client = OpenAI()

def main():
    file_ids = upload_data(
        training_file_name=setting.dataset_train_filename,
        validation_file_name=setting.dataset_val_filename,
    )


def upload_data(training_file_name, validation_file_name):
    training_response = client.files.create(
        file=open(training_file_name, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response.id

    validation_response = client.files.create(
        file=open(validation_file_name, "rb"), purpose="fine-tune"
    )
    validation_file_id = validation_response.id

    file_ids = {
        "training_file_id": training_file_id,
        "validation_file_id": validation_file_id,
    }
    print(file_ids)
    save_to_json(file_ids, setting.file_id_filename)
    return file_ids


if __name__ == "__main__":
    main()
