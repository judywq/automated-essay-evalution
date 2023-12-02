from time import sleep
from openai import OpenAI
from icecream import ic
from lib.io import save_to_json
import setting

client = OpenAI()


class FineTuningHelper:

    def run(self, training_file_name, validation_file_name):
        file_ids = self.upload_data(training_file_name, validation_file_name)
        job_id = self.start_training(
            training_file_id=file_ids["training_file_id"],
            validation_file_id=file_ids["validation_file_id"],
            suffix_name=setting.model_suffix,
        )
        self.wait_for_training_job(job_id=job_id)

    def upload_data(self, training_file_name, validation_file_name):
        train_file_obj = client.files.create(
            file=open(training_file_name, "rb"), purpose="fine-tune"
        )
        training_file_id = train_file_obj.id

        validation_file_obj = client.files.create(
            file=open(validation_file_name, "rb"), purpose="fine-tune"
        )
        validation_file_id = validation_file_obj.id

        file_ids = {
            "training_file_id": training_file_id,
            "validation_file_id": validation_file_id,
        }
        print(file_ids)
        save_to_json(file_ids, setting.file_id_filename)
        return file_ids

    def start_training(self, training_file_id, validation_file_id, suffix_name):
        # Create a Fine Tuning Job
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=setting.fine_tuning_base_model_id,
            suffix=suffix_name,
        )

        print(job)
        self.save_job(job)
        return job.id

    def list_jobs(self):
        jobs = client.fine_tuning.jobs.list()
        return jobs.data
    
    def wait_for_training_job(self, job_id):
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(job)
        self.save_job(job)

        while job.status not in ("succeeded", "failed", "cancelled"):
            event_resp = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=30)

            events = event_resp.data
            events.reverse()

            for event in events:
                ic(event.message)

            print("Waiting for 60 seconds...")
            sleep(60)
            job = client.fine_tuning.jobs.retrieve(job_id)
            print(job)

        self.save_job(job)
        
        return job.status == "succeeded"

    def save_job(self, job):
        data = dict(job)
        if 'hyperparameters' in data:
            del data['hyperparameters']
        save_to_json(data, setting.job_id_filename)
