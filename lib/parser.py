import json
from lib.essay import Essay

import logging
logger = logging.getLogger(__name__)


class ParserBase():
    """Parser is a class that is responsible for generating prompt and 
    parsing the response from ChatGPT.
    
    Returns:
        dict: {"success": bool, "raw_response": str, "result": str, ...}
    """
    failure = ""
    result_key = "result"
    task_name = "<Abstract>"
    
    def __init__(self):
        self.inputs = None
    
    def compose_prompt(self, inputs):
        """Compose the prompt for ChatGPT from inputs

        Args:
            inputs (dict): any inputs that are needed to compose the prompt

        Returns:
            str: prompt for ChatGPT
        """
        self.inputs = inputs
        return ""
    
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
            'success': success,
            'prompt': prompt,
            'raw_response': response,
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
    def remove_surrounding_quotes(text:str) -> str:
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

    def get_sample_response(self, prompt):
        return ""


class EssayEvaluationParser(ParserBase):
    """Parse the result of essay evaluation from ChatGPT
    """
    
    def compose_prompt(self, inputs):
        super().compose_prompt(inputs=inputs)
        system_message = inputs['system_message']
        essay: Essay = inputs['essay']
        return f"""{system_message}
Please return your evaluation and feedback in JSON format of {{ "level": ..., "reasoning": ...}}
The essay prompt is: `{essay.prompt_text}`
The essay is: `{essay.text}`
"""
    
    def parse_response(self, prompt, response):
        res = super().parse_response(prompt=prompt, response=response)
        essay: Essay = self.inputs['essay']
        try:
            obj = json.loads(response)
            
            if 'level' not in obj:
                logger.warning(f"Cannot find 'level' in response: {obj}")
            if 'reasoning' not in obj:
                logger.warning(f"Cannot find 'reasoning' in response: {obj}")
            
            return {
                **res,
                "result": {
                    "ok": obj.get('level', '') == essay.level.value,
                    "level": essay.level.value,
                    "level_resp": obj.get('level', ''),
                    "reasoning": obj.get('reasoning', ''),
                    "essay_prompt": essay.prompt,
                    "essay": essay.text,
                    "gpt_prompt": prompt,
                    "raw_response": response,
                    },
            }
        except json.decoder.JSONDecodeError as e:
            return {
                **res,
                "success": False,
            }
    


class RationalParser(ParserBase):
    """Test the rationality of several words in a sentence
    
    inputs={"words": ["account", "apple"], "sentence": "I have an ______ with the bank."}
    
    return {"success": True, "result": {"account": True, "apple": False}, "good_candidates": ["apple"], "others": ["account"], "words": ["account", "bank"], "sentence": "I have an ______ with the bank."}
    """
    task_name = "Rationality Test"
    
    def compose_prompt(self, inputs):
        super().compose_prompt(inputs=inputs)
        keyword = inputs.get('keyword')
        candidates = inputs.get('candidates')
        words_with_comma = ", ".join(set(str(w) for w in candidates))
        sentence = inputs.get('sentence')
        
        prompt = f'''You are an English teacher at a Japanese university and you are creating distractors for vocabulary multiple-choice cloze questions for your students. 
In this multiple choice cloze question stem: "{sentence}" 
A list of possible distractors include "{words_with_comma}". 
Please provide feedback in terms of syntactic appropriateness and contextual/semantic sense-making of the distractors in the completed sentences. 
Return only the following result in JSON format to me:
{{
  "word": {{"syntax": true, "semantics": true}},
  "word": {{"syntax": true, "semantics": false}}
}}

To provide you with more instructions on judgement, consider the the question stem: "Birds _____ in the sky." The list of distractors include "swim, beat". The distractor "swim" is syntactically valid because there will be no grammar errors when it is filled into the blank, but it does not make sense since birds "fly" in the sky, not "swim". The distractor "beat" is syntactically inappropriate because "beat" is a transitive verb and requires an object after it. There will be  grammar errors when it is filled into the blank. It also does not make much sense or is incomprehensible. Return only the following result in JSON format to me:
{{
  "swim": {{"syntax": true, "semantics": false}},
  "beat": {{"syntax": false, "semantics": false}}
}}'''
# Reply with json object only without any notes. 
# while using correct articles and prepositions, \

        return prompt

    def parse_response(self, prompt, response):
        res = super().parse_response(prompt=prompt, response=response)
        try:
            obj = json.loads(response)
            others = []
            good_candidates = []
            candidates = self.inputs['candidates']
            for k, v in obj.items():
                candidate = next(filter(lambda w: str(w) == k, candidates), None)
                if not candidate:
                    logger.warning(f"Cannot find candidate '{k}' in response: {candidates}")
                    continue
                if v['syntax'] and not v['semantics']:
                    # the word is a good candidate as a distractor 
                    #   if it is syntactically correct but semantically wrong
                    good_candidates.append(candidate)
                else:
                    others.append(candidate)
            return {
                **res,
                "result": obj,
                "good_candidates": good_candidates,
                "others": others,
            }
        except json.decoder.JSONDecodeError as e:
            return {
                **res,
                "success": False,
            }
        
    
    def get_sample_response(self, prompt):
        return {
            "success": True,
            "others": ["account"],
            "good_candidates": ["bank"],
            "response": {"account": True, "bank": False}, 
            "words": ["account", "bank"], 
            "sentence": "I have an ______ with the bank."
        }



