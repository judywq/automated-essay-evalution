import asyncio
import os
from lib.io import read_json, save_to_jsonl
from lib.api_request_google import process_api_requests_from_file


def prepare_data():
    json_file = "data/test/gemini/one.json"
    input_file = "data/test/gemini/one.jsonl"

    record = read_json(json_file)
    dataset = [record]
    save_to_jsonl(dataset, input_file)


def main():
    input_file = "data/test/gemini/one.jsonl"
    output_file = "data/test/gemini/one.result.jsonl"
    api_key = os.getenv("GOOGLE_API_KEY")
    request_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=input_file,
            save_filepath=output_file,
            request_url=request_url,
            api_key=api_key,
        )
    )

if __name__ == "__main__":
    # prepare_data()
    main()
