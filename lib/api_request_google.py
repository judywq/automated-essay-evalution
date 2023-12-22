import time
import aiohttp
import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field

# Constants
SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR = 15
SECONDS_TO_SLEEP_EACH_LOOP = 0.001
SECONDS_TO_SLEEP_EACH_LOOP = 1


@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0


@dataclass
class APIRequest:
    task_id: int
    request_json: dict
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(self, session, request_url, query_params, retry_queue, save_filepath, status_tracker):
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(url=request_url, params=query_params, json=self.request_json) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(f"Request {self.task_id} failed with error {response['error']}")
                status_tracker.num_api_errors += 1
                error = response.get("error", {})
                message = error.get("message", "")
                if "Quota exceeded" in message or "Resource has been exhausted" in message:
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1
        except Exception as e:
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
            message = error.get("message", "")
            if "Quota exceeded" in message or "Resource has been exhausted" in message:
                status_tracker.time_of_last_rate_limit_error = time.time()
                status_tracker.num_rate_limit_errors += 1
                status_tracker.num_api_errors -= 1

        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request failed after all attempts. Saving errors: {self.result}")
                data = [self.request_json, [str(e) for e in self.result], self.metadata] if self.metadata else [self.request_json, [str(e) for e in self.result]]
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = [self.request_json, response, self.metadata] if self.metadata else [self.request_json, response]
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


def append_to_jsonl(data, filename):
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


async def process_api_requests_from_file(requests_filepath, save_filepath, request_url, api_key, max_attempts=5, logging_level=logging.INFO, additional_params=None):
    logging.basicConfig(level=logging_level)
    # request_header = {"Authorization": f"Bearer {api_key}"}
    query_params = {"key": api_key}

    status_tracker = StatusTracker()
    queue_of_requests_to_retry = asyncio.Queue()
    next_request = None
    file_not_finished = True

    with open(requests_filepath) as file:
        requests = file.__iter__()
        async with aiohttp.ClientSession() as session:
            while True:
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                    elif file_not_finished:
                        try:
                            request_json = json.loads(next(requests))
                            request_json.update(additional_params)
                            next_request = APIRequest(task_id=status_tracker.num_tasks_started, request_json=request_json, attempts_left=max_attempts, metadata=request_json.pop("metadata", None))
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                        except StopIteration:
                            file_not_finished = False
                if next_request:
                    asyncio.create_task(next_request.call_api(session, request_url, query_params, queue_of_requests_to_retry, save_filepath, status_tracker))
                    next_request = None

                if status_tracker.num_tasks_in_progress == 0 and not file_not_finished:
                    break

                await asyncio.sleep(SECONDS_TO_SLEEP_EACH_LOOP)

                if time.time() - status_tracker.time_of_last_rate_limit_error < SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR:
                    await asyncio.sleep(SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR - (time.time() - status_tracker.time_of_last_rate_limit_error))

        logging.info(f"Parallel processing complete. Results saved to {save_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath")
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--request_url", default="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent")
    parser.add_argument("--api_key", default=os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")

    additional_params = {}
    if args.temperature is not None:
        additional_params["generationConfig"] = {"temperature": args.temperature}

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            request_url=args.request_url,
            api_key=args.api_key,
            max_attempts=args.max_attempts,
            logging_level=args.logging_level,
            additional_params=additional_params,
        )
    )
