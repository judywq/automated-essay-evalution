import os
import json
from time import sleep
import openai
from icecream import ic
from lib.utils import convert_essay, format_check
from lib.stats import print_stats
import setting


def main():
    resp = json.load(open(setting.job_id_filename, "r"))
    job_id = resp["id"]
    print(job_id)

    check_training(job_id)


def check_training(job_id):
    response = openai.FineTuningJob.retrieve(job_id)
    print(response)
    json.dump(response, open(setting.job_id_filename, "w"))

    while response["status"] in ("running", "validating_files"):
        event_resp = openai.FineTuningJob.list_events(id=job_id, limit=30)

        events = event_resp["data"]
        events.reverse()

        for event in events:
            ic(event["message"])

        print("Waiting for 60 seconds...")
        sleep(60)
        response = openai.FineTuningJob.retrieve(job_id)
        print(response)

    print(response)
    fine_tuned_model_id = response["fine_tuned_model"]
    print("\nFine-tuned model id:", fine_tuned_model_id)
    json.dump(response, open(setting.job_id_filename, "w"))
    return fine_tuned_model_id


if __name__ == "__main__":
    main()
