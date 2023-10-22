import json
import openai
from lib.utils import convert_essay, format_check
from lib.stats import print_stats
import setting


def main():
    file_ids = json.load(open(setting.uploaded_file_id_filename, "r"))
    print(file_ids)
    # start_training(
    #     training_file_id=file_ids["training_file_id"],
    #     validation_file_id=file_ids["validation_file_id"],
    # )


def start_training(training_file_id, validation_file_id, suffix_name):
    # Create a Fine Tuning Job
    response = openai.FineTuningJob.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="gpt-3.5-turbo",
        suffix=suffix_name,
    )

    job_id = response["id"]

    print(response)

    response = openai.FineTuningJob.retrieve(job_id)
    print(response)

    response = openai.FineTuningJob.list_events(id=job_id, limit=50)

    events = response["data"]
    events.reverse()

    for event in events:
        print(event["message"])

    response = openai.FineTuningJob.retrieve(job_id)
    fine_tuned_model_id = response["fine_tuned_model"]
    print(response)
    print("\nFine-tuned model id:", fine_tuned_model_id)
    json.dump(response, open(setting.model_id_filename, "w"))
    return fine_tuned_model_id



def run_model(fine_tuned_model_id):
    # Generating using the new model
    test_messages = []
    test_messages.append({"role": "system", "content": setting.system_message})
    user_message = "How are you today Samantha"
    test_messages.append({"role": "user", "content": user_message})

    print(test_messages)

    response = openai.ChatCompletion.create(
        model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=500
    )
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
