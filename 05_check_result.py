import os
import json
import openai
from icecream import ic
from lib.utils import convert_essay, format_check
from lib.stats import print_stats
import setting


def main():
    job_id_filename = os.path.join(setting.output_root, 'ids', 'job-id-eassy-test-1-2023-10-22-19-24-49.json')
    resp = json.load(open(job_id_filename, "r"))
    job_id = resp["id"]
    print(job_id)
    check_training(job_id)


def check_training(job_id):
    response = openai.FineTuningJob.retrieve(job_id)
    print(response)

    response = openai.FineTuningJob.list_events(id=job_id, limit=50)

    events = response["data"]
    events.reverse()

    for event in events:
        ic(event["message"])
    
    return

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
