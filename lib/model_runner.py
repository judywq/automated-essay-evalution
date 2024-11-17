import os
import asyncio
import logging
from lib.finetuning import FineTuningHelper
from lib.api_request_parallel_processor import process_api_requests_from_file_openai
from lib.api_request_google import process_api_requests_from_file as process_api_requests_from_file_google

logger = logging.getLogger(__name__)

class ModelRunner:
    def __init__(self, config) -> None:
        self.config = config
        
    def run(self, skip_if_exists=True):
        self._run_openai_finetuned(skip_if_exists=skip_if_exists)
        self._run_baseline_models(skip_if_exists=skip_if_exists)

    def _run_openai_finetuned(self, skip_if_exists=True):
        if not self.config.run_finetuned:
            return
        finetuner = FineTuningHelper(self.config)
        job = finetuner.try_load_job()
        if job and job['status'] == 'succeeded':
            fine_tuned_model = job['fine_tuned_model']
            input_fn = self.config.dataset_test_short_filename
            output_fn = self.config.get_dataset_test_result_finetuned_filename()
            
            if skip_if_exists and os.path.exists(output_fn):
                logger.info(f"Skip running model {fine_tuned_model}.")
            else:
                self._run_openai_model(
                    input_jsonl_fn=input_fn,
                    output_jsonl_fn=output_fn,
                    model=fine_tuned_model,
                    temperature=self.config.temperature,
                )
        else:
            logger.info(f"Fine-tuning model is not ready yet: {job}")


    def _run_baseline_models(self, skip_if_exists=True):
        if not self.config.run_baseline:
            return
        for data_path in self.config.data_paths:
            if not data_path.active or data_path.is_finetuned:
                continue
            model_id = data_path.llm_model_id
            input_fn = data_path.dataset_in
            output_fn = data_path.dataset_out
            
            if skip_if_exists and os.path.exists(output_fn):
                logger.info(f"Skip running model {model_id}.")
                continue
            
            if data_path.format == 'openai':
                self._run_openai_model(
                    input_jsonl_fn=input_fn,
                    output_jsonl_fn=output_fn,
                    model=model_id,
                    temperature=self.config.temperature,
                )
            elif data_path.format == 'gemini':
                self._run_google_model(
                    input_jsonl_fn=input_fn,
                    output_jsonl_fn=output_fn,
                    model=model_id,
                    temperature=self.config.temperature,
                )
            else:
                raise Exception(f"Unknown model format: {data_path.format}")

    @classmethod
    def _run_openai_model(
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
            process_api_requests_from_file_openai(
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

    @classmethod
    def _run_google_model(
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

        additional_params = {}
        # if model is not None:
            # additional_params["model"] = model
        if temperature is not None:
            additional_params["generationConfig"] = {"temperature": temperature}

        api_key = os.getenv("GOOGLE_API_KEY")
        request_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        asyncio.run(
            process_api_requests_from_file_google(
                requests_filepath=input_jsonl_fn,
                save_filepath=output_jsonl_fn,
                request_url=request_url,
                api_key=api_key,
                max_attempts=max_attempts,
                logging_level=logging_level,
                additional_params=additional_params,
            )
        )

