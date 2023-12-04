import os
import asyncio
import logging
from lib.finetuning import FineTuningHelper
from lib.api_request_parallel_processor import process_api_requests_from_file

logger = logging.getLogger(__name__)

class ModelRunner:
    def __init__(self, config) -> None:
        self.config = config

    def run(self, skip_if_exists=True):
        finetuner = FineTuningHelper(self.config)
        job = finetuner.try_load_job()
        if job and job['status'] == 'succeeded':
            fine_tuned_model = job['fine_tuned_model']
            input_fn = self.config.dataset_test_short_filename
            output_fn = self.config.get_dataset_test_result_finetuned_filename()
            
            if skip_if_exists and os.path.exists(output_fn):
                logger.info(f"Skip running model {fine_tuned_model}.")
            else:
                self._run_model(
                    input_jsonl_fn=input_fn,
                    output_jsonl_fn=output_fn,
                    model=fine_tuned_model,
                    temperature=0,
                )
        else:
            logger.info(f"Fine-tuning model is not ready yet: {job}")
        
        for baseline_model in self.config.baseline_models:
            model_id = baseline_model["id"]
            input_fn = self.config.dataset_test_full_filename
            output_fn = self.config.get_dataset_test_result_filename(
                input_fn=input_fn,
                model_name=model_id,
            )
            
            if skip_if_exists and os.path.exists(output_fn):
                logger.info(f"Skip running model {model_id}.")
                continue
            self._run_model(
                input_jsonl_fn=input_fn,
                output_jsonl_fn=output_fn,
                model=model_id,
                temperature=0,
            )

    @classmethod
    def _run_model(
        cls,
        input_jsonl_fn,
        output_jsonl_fn,
        model=None,
        temperature=0,
        max_attempts=5,
        logging_level=logging.INFO,
    ):
        logger.info(f"Run model [{model}] with input: {input_jsonl_fn}.")
        # If model and temperature are None, the value in the input file will be used.
        api_key = os.getenv("OPENAI_API_KEY")
        request_url = "https://api.openai.com/v1/chat/completions"
        max_requests_per_minute = 3_000 * 0.5
        max_tokens_per_minute = 250_000 * 0.5
        token_encoding_name = "cl100k_base"

        additional_params = {}
        if model is not None:
            additional_params["model"] = model
        if temperature is not None:
            additional_params["temperature"] = temperature

        # run script
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=input_jsonl_fn,
                save_filepath=output_jsonl_fn,
                request_url=request_url,
                api_key=api_key,
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute,
                token_encoding_name=token_encoding_name,
                max_attempts=max_attempts,
                logging_level=logging_level,
                additional_params=additional_params,
            )
        )
