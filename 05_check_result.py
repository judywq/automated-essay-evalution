import os
import json
from time import sleep
from openai import OpenAI
from icecream import ic
from lib.io import save_to_json
import setting


client = OpenAI()

def main():
    resp = json.load(open(setting.job_id_filename, "r"))
    job_id = resp["id"]
    print(job_id)

    check_training(job_id)


def check_training(job_id):
    response = client.fine_tuning.jobs.retrieve(job_id)
    print(response)
    save_to_json(response, setting.job_id_filename)

    while response.status in ("running", "validating_files"):
        event_resp = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=30)

        events = event_resp.data
        events.reverse()

        for event in events:
            ic(event.message)

        print("Waiting for 60 seconds...")
        sleep(60)
        response = client.fine_tuning.jobs.retrieve(job_id)
        print(response)

    print(response)
    fine_tuned_model_id = response.fine_tuned_model
    print("\nFine-tuned model id:", fine_tuned_model_id)
    save_to_json(response, setting.job_id_filename)
    return fine_tuned_model_id


if __name__ == "__main__":
    main()
