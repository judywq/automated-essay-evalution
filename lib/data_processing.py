import os
import re
import json
from collections import defaultdict
import pandas as pd
from lib.io import read_data, write_data
from lib.essay import Essay
from lib.utils import calc_agreement, calc_success_rate_dict
from lib.io import save_to_jsonl

import logging
logger = logging.getLogger(__name__)


class DataSplitter:
    
    def __init__(self, config) -> None:
        self.config = config
        self.max_num_per_group = {}
    
    def run(self, skip_if_exist=True):
        if skip_if_exist and self.all_files_exist():
            print("Data splits already exist, skip.")
            return
        df = self.read_all_data()
        self.split_data(df, train_on_form=self.config.train_on_form)
        print("Data splitting done!")
        
    
    def all_files_exist(self):
        for fn in [
            self.config.index_train_filename,
            self.config.index_test_filename,
            self.config.index_val_filename,
        ]:
            if not os.path.exists(fn):
                return False
        return True

    def read_all_data(self):
        form1_fn = self.config.index_file_path_template.format(form_id=1)
        form2_fn = self.config.index_file_path_template.format(form_id=2)

        df1 = read_data(form1_fn)
        df2 = read_data(form2_fn)
        
        self.calc_max_num_per_group(df1, df2)

        df1[self.config.column_form] = 1
        df2[self.config.column_form] = 2
        df = pd.concat([df1, df2], ignore_index=True)
        
        df[self.config.column_item] = self.config.item_id
                
        logger.info("Score shape:\n{}".format(df.value_counts([self.config.column_form, self.config.column_score]).sort_index()))

        return df
    
    def calc_max_num_per_group(self, df1, df2):
        value_counts_dict_df1 = df1[self.config.column_score].value_counts().to_dict()
        value_counts_dict_df2 = df2[self.config.column_score].value_counts().to_dict()

        # Combine dictionaries and calculate the minimum value for each key
        self.max_num_per_group = {key: min(value_counts_dict_df1.get(key, 0), value_counts_dict_df2.get(key, 0))
                        for key in set(value_counts_dict_df1) | set(value_counts_dict_df2)}
        
        logger.debug(f"Max num per group: {self.max_num_per_group}")

    def split_data(self, df, train_on_form=None):
        """Split data into train/val/test sets

        Args:
            df (pd.DataFrame): input data
            train_on_form (int, optional): train only on form N. Default (None) means both .
        """
        filter_integer = self.config.integer_score_only
        df_copy = df.copy()
        df_train = self.sample(df_copy, num_per_group=self.config.num_per_group["train"], filter_form=train_on_form, filter_integer=filter_integer)
        df_copy.drop(df_train.index, inplace=True)
        df_val = self.sample(df_copy, num_per_group=self.config.num_per_group["val"], filter_form=train_on_form, filter_integer=filter_integer)
        df_copy.drop(df_val.index, inplace=True)
        df_test = self.sample(df_copy, num_per_group=self.config.num_per_group["test"])

        write_data(df_train, self.config.index_train_filename)
        write_data(df_test, self.config.index_test_filename)
        write_data(df_val, self.config.index_val_filename)
        
        print("Train shape:", df_train.shape)
        print("Val shape:", df_val.shape)
        print("Test shape:", df_test.shape)
    
    def sample(self, df: pd.DataFrame, num_per_group=-1, filter_form=None, filter_integer=False) -> pd.DataFrame:
        
        if filter_integer:
            df = df[df[self.config.column_score] % 1 == 0].copy()

        if filter_form:
            df = df[df[self.config.column_form] == filter_form]
        
        if num_per_group < 0:
            return df

        sampled_data = pd.concat([group.sample(min(num_per_group, self.max_num_per_group[key], len(group))) 
                                  for key, group in df.groupby(self.config.column_score)])
        return sampled_data


class DatasetPreparation:
    
    def __init__(self, config) -> None:
        self.config = config

    def run(self, skip_if_exist=True):
        # Training data
        self.prepare(
            input_file=self.config.index_train_filename,
            system_message=self.config.system_message_full,
            dataset_fn=self.config.dataset_train_filename,
            skip_if_exist=skip_if_exist,
            for_training=True,
        )
        # Validation data
        self.prepare(
            input_file=self.config.index_val_filename,
            system_message=self.config.system_message_full,
            dataset_fn=self.config.dataset_val_filename,
            skip_if_exist=skip_if_exist,
            for_training=True,
        )
        # Test data for official models
        self.prepare(
            input_file=self.config.index_test_filename,
            system_message=self.config.system_message_full,
            dataset_fn=self.config.dataset_test_full_filename,
            skip_if_exist=skip_if_exist,
            for_training=False,
        )
        # Test data for fine-tuned model
        self.prepare(
            input_file=self.config.index_test_filename,
            system_message=self.config.system_message_short,
            dataset_fn=self.config.dataset_test_short_filename,
            skip_if_exist=skip_if_exist,
            for_training=False,
        )

    def prepare(self, input_file, system_message, dataset_fn, chunk_size=1, skip_if_exist=True, for_training=True):
        if skip_if_exist and os.path.exists(dataset_fn):
            logger.debug(f"Dataset {dataset_fn} already exists, skip.")
            return
        essay_list = Essay.load_essays_from_file(input_file)
        dataset = []
        for chunk in self.divide_chunks(essay_list, chunk_size):
            record = self.convert_essay_to_dataset(chunk, system_message, for_training)
            dataset.append(record)
        save_to_jsonl(dataset, dataset_fn)

        # Initial dataset stats
        logger.info(f"Num of records: [{len(dataset)}] in [{dataset_fn}]")
        format_errors = self.format_check(dataset, for_training=for_training)
        if format_errors:
            logger.error(f"Found errors: {format_errors}")
            return

        # print_stats(dataset)

    @classmethod
    def convert_essay_to_dataset(cls, essays: list[Essay], system_message=None, for_training=True) -> dict:
        # Initializing the messages list
        messages = []

        if for_training:
            # Including the system message if provided
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            # Iterating through the lines and formatting the messages
            for essay in essays:
                # Formatting the message
                message = {
                    "role": "user",
                    "content": cls.prompt_formatter_short(essay)
                }
                messages.append(message)
                message = {
                    "role": "assistant",
                    "content": cls.score_formatter(essay.score)
                }
                messages.append(message)
            # Creating the final output dictionary
            output_dict = {
                "messages": messages,
            }        
            return output_dict        
        else:
            # Iterating through the lines and formatting the messages
            for essay in essays:
                # Formatting the message
                message = {
                    "role": "user",
                    "content": cls.prompt_formatter_full(essay, system_message=system_message)
                }
                messages.append(message)
                
            # Creating the final output dictionary
            output_dict = {
                "messages": messages,
                "metadata": { "essay": essay.to_dict()},
            }
            return output_dict

    @staticmethod
    def prompt_formatter_short(essay: Essay):
        return f"""Essay prompt: `{essay.prompt_text}`
    Essay content: `{essay.text}`"""
    
    @staticmethod
    def prompt_formatter_full(essay: Essay, system_message):
        user_content = f"""{system_message}
Please return your evaluation and feedback in JSON format of {{ "score": ..., "reasoning": ...}}.
Please do not include markdown formatting in your response.
The essay prompt is: `{essay.prompt_text}`
The essay is: `{essay.text}`
"""
        return user_content

    @staticmethod
    def score_formatter(score):
        return f'{score:.1f}'

    @staticmethod
    def format_check(dataset, for_training=True):    
        # Format error checks
        format_errors = defaultdict(int)

        for ex in dataset:
            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue

            messages = ex.get("messages", None)
            if not messages:
                format_errors["missing_messages_list"] += 1
                continue

            for message in messages:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1

                if any(k not in ("role", "content", "name") for k in message):
                    format_errors["message_unrecognized_key"] += 1

                if message.get("role", None) not in ("system", "user", "assistant"):
                    format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                if not content or not isinstance(content, str):
                    format_errors["missing_content"] += 1

            if for_training:        
                if not any(message.get("role", None) == "assistant" for message in messages):
                    format_errors["example_missing_assistant_message"] += 1
        return format_errors

    @staticmethod
    def divide_chunks(l, n): 
    # Yield successive n-sized 
    # chunks from l. 
        
        # looping till length l 
        for i in range(0, len(l), n):  
            yield l[i:i + n] 


class ResponseParser:
    
    def __init__(self, config) -> None:
        self.config = config

    def run(self, skip_if_exist=True):
        for dp in self.config.data_paths:
            self.parse_response(
                input_file=dp.dataset_out,
                output_file=dp.result_file,
                skip_if_exist=skip_if_exist,
            )
            
    def parse_response(self, input_file, output_file, skip_if_exist=True):
        if not os.path.exists(input_file):
            logger.error(f"Input file {input_file} does not exist.")
            return
        if skip_if_exist and os.path.exists(output_file):
            logger.info(f"Result file {output_file} already exists, skip parsing.")
            return
        rows = []
        with open(input_file, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                for message in json_data[0]["messages"]:
                    if message["role"] == "user":
                        gpt_prompt = message["content"]
                        break
                raw_response = json_data[1]["choices"][0]["message"]["content"]
                res = self.parse_raw_response(raw_response)
                essay_data = json_data[2]["essay"]
                agreement = calc_agreement(ground_truth_score=essay_data["ETS Score"], gpt_score=res["score"])
                row_data = {
                    **agreement,
                    "GPT Score": res["score"],
                    **essay_data,
                    "gpt_prompt": gpt_prompt,
                    "raw_response": raw_response,
                    "reasoning": res["reasoning"],
                }
                rows.append(row_data)
        df = pd.DataFrame(rows)
        write_data(df, output_file)
    
    @classmethod
    def parse_raw_response(cls, raw_response):
        score_pattern = re.compile(r'"score":\s*"?(\d+(?:\.\d+)?)"?')
        reasoning_pattern = re.compile(r'"reasoning":\s*"(.*)"', re.DOTALL)
        
        raw_response = cls.remove_surrounding_apostrophe(raw_response)
        try:
            res = json.loads(raw_response)
            res["score"] = float(res["score"])
        except json.decoder.JSONDecodeError as e:
            match = score_pattern.findall(raw_response)
            score = -1
            if match:
                score = float(match[0])
            else:
                logger.error(f"Cannot parse response \n {raw_response}")
                res = {
                    "score": -1,
                    "reasoning": "",
                }
                return res

            match = reasoning_pattern.findall(raw_response)
            reasoning = ""
            if match:
                reasoning = match[0]
            res = {
                "score": score,
                "reasoning": reasoning,
            }
        return res

    @classmethod
    def remove_surrounding_apostrophe(cls, text: str) -> str:
        """Remove surrounding apostrophe from a string

        Args:
            text (str): string possibly with surrounding apostrophe, such as "```json ... ```"

        Returns:
            str: string without surrounding apostrophe
        """
        if text.startswith("```json\n"):
            text = text[8:]
        if text.endswith("```"):
            text = text[:-3]
        return text


class SummaryGenerator:
    
    def __init__(self, config) -> None:
        self.config = config

    def run(self, skip_if_exist=True):
        def combine_labels(rate_label, model_label):
            return f"{rate_label}:{model_label}"
        res = {
            'label': self.config.run_prefix,
            'train_on_form': self.config.tof_name,
            'integer_score_only': self.config.integer_score_only,
            'train_size': self.calc_dataset_size(self.config.index_train_filename),
            'val_size': self.calc_dataset_size(self.config.index_val_filename),
            'test_size': self.calc_dataset_size(self.config.index_test_filename),
        }
        model_labels = [dp.llm_model_label for dp in self.config.data_paths]
        rate_labels = []
        tmp = {}
        for dp in self.config.data_paths:
            rate_dict = calc_success_rate_dict(dp.result_file, self.config.integer_score_only)
            for rate_label, rate in rate_dict.items():
                if rate_label not in rate_labels:
                    rate_labels.append(rate_label)
                field_name = combine_labels(rate_label, dp.llm_model_label)
                tmp[field_name] = rate
        
        # Order the fields so that rate_labels are in higher priority
        for rate_label in rate_labels:
            for model_label in model_labels:
                field_name = combine_labels(rate_label, model_label)
                res[field_name] = tmp[field_name]
        write_data(pd.DataFrame([res]), self.config.result_summary_filename)
        
        return res
        
    def calc_dataset_size(self, fn):
        df = read_data(fn)
        return df.shape[0]