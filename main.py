import json
import openai
from lib.utils import convert_conversation, format_check
from lib.stats import print_stats

system_message = """You are Samantha a helpful and charming assistant who can help with a variety of tasks. You are friendly and often flirt"""

def main():
    # I am picking one file here but you would probably want to do a lot more for a proper model
    data_path = "./data/samantha-data/data/howto_conversations.jsonl"
    # Load dataset
    with open(data_path) as f:
        json_dataset = [json.loads(line) for line in f]    

    dataset = []
    for data in json_dataset:
        record = convert_conversation(data, system_message=system_message)
        dataset.append(record)

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)

    format_errors = format_check(dataset)
    if format_errors:
        print(f"Found errors: {format_errors}")
        return

    print_stats(dataset)
    


# Upload your data
# curl -https://api.openai.com/v1/files \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -F "purpose=fine-tune" \
#   -F "file=@path_to_your_file"


def upload_data(training_file_name, validation_file_name):
    training_response = openai.File.create(
        file=open(training_file_name, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response["id"]

    validation_response = openai.File.create(
        file=open(validation_file_name, "rb"), purpose="fine-tune"
    )
    validation_file_id = validation_response["id"]

    file_ids =  {
        "training_file_id": training_file_id,
        "validation_file_id": validation_file_id,
    }
    print(file_ids)
    return file_ids


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


def run_model(fine_tuned_model_id):

    # Generating using the new model
    test_messages = []
    test_messages.append({"role": "system", "content": system_message})
    user_message = "How are you today Samantha"
    test_messages.append({"role": "user", "content": user_message})

    print(test_messages)

    response = openai.ChatCompletion.create(
        model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=500
    )
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
