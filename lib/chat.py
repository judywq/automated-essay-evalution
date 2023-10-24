import openai
from tenacity import retry, stop_after_attempt
import setting

import logging

logger = logging.getLogger(__name__)


class MyBotWrapper:
    def __init__(self, parser, model=setting.DEFAULT_MODEL, temperature=0.5) -> None:
        self.parser = parser
        self.model = model
        self.temperature = temperature

    @retry(stop=stop_after_attempt(3))
    def run(self, inputs):
        prompt = self.parser.compose_prompt(inputs=inputs)
        logger.debug(f"PROMPT: {prompt}")
        response = self.get_completion(
            user_content=prompt.get("user_content"),
            system_content=prompt.get("system_content")
        )
        logger.debug(f"RAW RESPONSE: {response}")
        res = self.parser.parse_response(prompt=prompt, response=response)
        logger.debug(f"PARSED RESPONSE: {res}")
        return res

    def get_completion(self, user_content, system_content=None):
        messages = []
        system_content and messages.append(
            {"role": "system", "content": system_content}
        )
        messages.append({"role": "user", "content": user_content})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,  # this is the degree of randomness of the model's output
            request_timeout=setting.REQUEST_TIMEOUT_SECS,
        )
        return response.choices[0].message["content"]

    @property
    def task_name(self):
        return self.parser.task_name if self.parser else ""
