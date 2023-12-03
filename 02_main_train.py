from lib.utils import setup_log
from lib.finetuning import FineTuningHelper
from lib.config import MyConfig
from icecream import ic
from configlist import config_list


def main():
    running_count = 0
    for config_files in config_list:
        config = MyConfig(file_paths=config_files)

        finetuner = FineTuningHelper(config)
        finetuner.run()
        
        job = finetuner.retrieve_job()
        if job:
            ic(job.status, job.id)
            if job.status in ["running", "validating_files"]:
                running_count += 1
        
        if running_count >= 3:
            ic("Too many running jobs, quit.")
            break


if __name__ == "__main__":
    setup_log()
    main()
