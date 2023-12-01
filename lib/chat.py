from openai import OpenAI
from tenacity import retry, stop_after_attempt
import setting

import logging

logger = logging.getLogger(__name__)

client = OpenAI()


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
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,  # this is the degree of randomness of the model's output
            timeout=setting.REQUEST_TIMEOUT_SECS,
            
            # comment out for now, not supported for gpt-3.5
            # response_format={ "type": self.parser.response_format },
        )
        return response.choices[0].message.content

    @property
    def task_name(self):
        return self.parser.task_name if self.parser else ""
