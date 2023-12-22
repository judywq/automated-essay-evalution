import json
import re
from lib.essay import Essay
from lib.data_processing import DatasetPreparation
from lib.utils import calc_agreement

import logging

logger = logging.getLogger(__name__)


class ParserBase:
    """Parser is a class that is responsible for generating prompt and
    parsing the response from ChatGPT.

    Returns:
        dict: {"success": bool, "raw_response": str, "result": str, ...}
    """

    failure = ""
    result_key = "result"
    task_name = "<Abstract>"
    response_format = "text"  # "text" | "json_object"

    def __init__(self):
        self.inputs = None

    def compose_prompt(self, inputs):
        """Compose the prompt for ChatGPT from inputs

        Args:
            inputs (dict): any inputs that are needed to compose the prompt

        Returns:
            dict: prompt for ChatGPT
        """
        self.inputs = inputs
        return {"system_content": "", "user_content": ""}

    def parse_response(self, prompt, response):
        """Parse the response from ChatGPT into desired format

        Args:
            prompt (str): prompt given to ChatGPT
            response (str): raw response from ChatGPT

        Returns:
            dict: parsed response
        """
        success = not self.response_failed(response=response)
        return {
            "success": success,
            "prompt": prompt,
            "raw_response": response,
            self.result_key: response,
            **self.inputs,
        }

    def response_failed(self, response):
        error_list = [
            "Failed to read response from ChatGPT",
            "against OpenAI's content policy",
        ]
        for err_msg in error_list:
            if err_msg in response.lower():
                logger.error(response)
                return True
        return False

    @staticmethod
    def remove_surrounding_quotes(text: str) -> str:
        """Remove surrounding quotes from a string

        Args:
            text (str): string possibly with surrounding quotes

        Returns:
            str: string without surrounding quotes
        """
        if text.startswith('"'):
            text = text[1:]
        if text.endswith('"'):
            text = text[:-1]
        return text

    @staticmethod
    def remove_surrounding_apostrophe(text: str) -> str:
        """Remove surrounding apostrophe from a string

        Args:
            text (str): string possibly with surrounding apostrophe, such as "```json ... ```"

        Returns:
            str: string without surrounding apostrophe
        """
        # TODO: replace this with regex
        if text.startswith("```json\n"):
            text = text[8:]
        if text.endswith("```"):
            text = text[:-3]
        return text

    def get_sample_response(self, prompt):
        return ""


class EssayEvaluationParser(ParserBase):
    """Parse the result of essay evaluation from official models"""

    response_format = "json_object"

    def compose_prompt(self, inputs):
        super().compose_prompt(inputs=inputs)
        system_message = inputs["system_message"]
        essay: Essay = inputs["essay"]
        user_content = f"""{system_message}
Please return your evaluation and feedback in JSON format of {{ "level": ..., "reasoning": ...}}.
Please do not include markdown formatting in your response.
The essay prompt is: `{essay.prompt_text}`
The essay is: `{essay.text}`
"""
        return {
            "user_content": user_content,
        }

    def parse_response(self, prompt, response):
        res = super().parse_response(prompt=prompt, response=response)
        essay: Essay = self.inputs["essay"]
        try:
            obj = json.loads(response)

            if "level" not in obj:
                logger.warning(f"Cannot find 'level' in response: {obj}")
            if "reasoning" not in obj:
                logger.warning(f"Cannot find 'reasoning' in response: {obj}")

            return {
                **res,
                "result": {
                    "ok": obj.get("level", "") == essay.level.value,
                    "level": essay.level.value,
                    "level_resp": obj.get("level", ""),
                    "reasoning": obj.get("reasoning", ""),
                    "filename": essay.fn,
                    "essay_form_id": essay.form_id,
                    "essay_item_id": essay.item_id,
                    "essay_prompt": essay.prompt,
                    "essay": essay.text,
                    "llm_prompt": prompt["user_content"],
                    "raw_response": response,
                },
            }
        except json.decoder.JSONDecodeError as e:
            return {
                **res,
                "success": False,
            }


class EssayEvaluationScoreParser(ParserBase):
    """Parse the result of essay evaluation from official models"""

    response_format = "json_object"  # "text" | "json_object"

    def compose_prompt(self, inputs):
        super().compose_prompt(inputs=inputs)
        system_message = inputs["system_message"]
        essay: Essay = inputs["essay"]
        user_content = f"""{system_message}
Please return your evaluation and feedback in JSON format of {{ "score": ..., "reasoning": ...}}.
Please do not include markdown formatting in your response.
The essay prompt is: `{essay.prompt_text}`
The essay is: `{essay.text}`
"""
        return {
            "user_content": user_content,
        }

    def parse_response(self, prompt, response):
        res = super().parse_response(prompt=prompt, response=response)
        essay: Essay = self.inputs["essay"]
        try:
            response = self.remove_surrounding_apostrophe(response)
            obj = json.loads(response)

            if "score" not in obj:
                logger.warning(f"Cannot find 'score' in response: {obj}")
            if "reasoning" not in obj:
                logger.warning(f"Cannot find 'reasoning' in response: {obj}")

            llm_score = obj.get("score", -1)
            agreement = calc_agreement(
                ground_truth_score=essay.score, llm_score=llm_score
            )

            return {
                **res,
                "result": {
                    **agreement,
                    "ETS Score": essay.score,
                    "LLM Score": llm_score,
                    "reasoning": obj.get("reasoning", ""),
                    "filename": essay.fn,
                    "essay_form_id": essay.form_id,
                    "essay_item_id": essay.item_id,
                    "essay_prompt": essay.prompt_text,
                    "essay": essay.text,
                    "llm_prompt": prompt["user_content"],
                    "raw_response": response,
                },
            }
        except json.decoder.JSONDecodeError as e:
            return {
                **res,
                "success": False,
                "error_msg": 'Invalid JSON format.',
            }


class EssayEvaluationWithTunedModelParser(ParserBase):
    """Parse the result of essay evaluation from fine-tuned model"""

    pat = re.compile(f"(high|medium|low|none)")

    def compose_prompt(self, inputs):
        super().compose_prompt(inputs=inputs)
        system_content = inputs["system_message"]
        essay: Essay = inputs["essay"]
        user_content = DatasetPreparation.prompt_formatter_short(essay)
        return {
            "system_content": system_content,
            "user_content": user_content,
        }

    def parse_response(self, prompt, response):
        res = super().parse_response(prompt=prompt, response=response)
        essay: Essay = self.inputs["essay"]
        try:
            llm_score = float(response)
        except ValueError:
            llm_score = -1

        agreement = calc_agreement(ground_truth_score=essay.score, llm_score=llm_score)

        return {
            **res,
            "result": {
                **agreement,
                "ETS Score": essay.score,
                "LLM Score": llm_score,
                "filename": essay.fn,
                "essay_form_id": essay.form_id,
                "essay_item_id": essay.item_id,
                "essay_prompt": essay.prompt_text,
                "essay": essay.text,
                "raw_response": response,
            },
        }

