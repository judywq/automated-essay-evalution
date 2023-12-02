import json
import setting

from lib.finetuning import FineTuningHelper


def list_jobs():
    helper = FineTuningHelper()
    jobs = helper.list_jobs()
    job=jobs[0]
    print(job)
    helper.save_job(job)
    

def main():
    resp = json.load(open(setting.job_id_filename, "r"))
    job_id = resp["id"]
    print(job_id)

    helper = FineTuningHelper()
    helper.wait_for_training_job(job_id=job_id)


if __name__ == "__main__":
    main()
    # list_jobs()
