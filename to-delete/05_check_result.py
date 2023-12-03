import json
import setting
from lib.finetuning import FineTuningHelper

import logging
logger = logging.getLogger(__name__)


def list_jobs():
    helper = FineTuningHelper()
    jobs = helper.list_jobs()
    job=jobs[0]
    print(job)
    helper.save_job(job)
    

def main():
    resp = json.load(open(setting.job_id_filename, "r"))
    job_id = resp["id"]
    logger.info(f"Loaded job_id: {job_id}")

    helper = FineTuningHelper()
    succeeded = helper.wait_for_training_job(job_id=job_id)
    if succeeded:
        logger.info(f"Training succeeded: {job_id}")
    else:
        logger.info(f"Training failed: {job_id}")


if __name__ == "__main__":
    main()
    # list_jobs()
