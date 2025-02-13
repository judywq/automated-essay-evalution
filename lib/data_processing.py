import os
import math
import re
import json
from collections import defaultdict
import pandas as pd
from lib.io import read_data, write_data
from lib.essay import Essay
from lib.utils import calc_agreement, calc_metrics_dict
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
        logger.info(f"Total number of samples: {df.shape[0]}")
        if self.config.split_algorithm == 1:
            self.split_data_1(df, train_on_form=self.config.train_on_form)
        elif self.config.split_algorithm == 2:
            self.split_data_2(df, train_on_form=self.config.train_on_form)
        elif self.config.split_algorithm == 3:
            self.split_data_3(df, train_on_form=self.config.train_on_form)
        else:
            raise Exception(f"Invalid split algorithm: {self.config.split_algorithm}, please check your config file.")
            
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
        
        score_distribution = df.value_counts([self.config.column_form, self.config.column_score]).sort_index()
        write_data(score_distribution, self.config.distribution_filename, index=True)
        logger.info("Score shape:\n{}".format(score_distribution))

        return df
    
    def calc_max_num_per_group(self, df1, df2):
        """ Calculate the maximum number of samples we could take for each score group"""
        
        value_counts_dict_df1 = df1[self.config.column_score].value_counts().to_dict()
        value_counts_dict_df2 = df2[self.config.column_score].value_counts().to_dict()

        # Combine dictionaries and calculate the minimum value for each key
        self.max_num_per_group = {key: min(value_counts_dict_df1.get(key, 0), value_counts_dict_df2.get(key, 0))
                        for key in set(value_counts_dict_df1) | set(value_counts_dict_df2)}
        
        logger.info(f"Max num per group: {self.max_num_per_group}")

    def split_data_1(self, df, train_on_form=None):
        """Split data into train/val/test sets

        Args:
            df (pd.DataFrame): input data
            train_on_form (int, optional): train only on form N. Default (None) means both .
        """
        integer_score_only = self.config.integer_score_only
        df_copy = df.copy()
        df_train = self.sample_1(df_copy, num_per_group=self.config.num_per_group["train"], filter_form=train_on_form, integer_score_only=integer_score_only)
        df_copy.drop(df_train.index, inplace=True)
        df_val = self.sample_1(df_copy, num_per_group=self.config.num_per_group["val"], filter_form=train_on_form, integer_score_only=integer_score_only)
        df_copy.drop(df_val.index, inplace=True)
        df_test = self.sample_1(df_copy, num_per_group=self.config.num_per_group["test"])
        
        self.print_distribution(df_train, df_val, df_test)

        write_data(df_train, self.config.index_train_filename)
        write_data(df_test, self.config.index_test_filename)
        write_data(df_val, self.config.index_val_filename)
    
    def sample_1(self, df: pd.DataFrame, num_per_group=-1, filter_form=None, integer_score_only=False) -> pd.DataFrame:
        if integer_score_only:
            df = df[df[self.config.column_score] % 1 == 0].copy()

        if filter_form:
            df = df[df[self.config.column_form] == filter_form]
        
        if num_per_group < 0:
            return df

        sampled_data = pd.concat([group.sample(min(num_per_group, self.max_num_per_group[key], len(group))) 
                                  for key, group in df.groupby(self.config.column_score)])
        return sampled_data

    def split_data_2(self, df, train_on_form=None):
        """Split data into train/val/test sets
            1. prompt1全部拿来做training/val(9:1)，prompt2做testing。
            2. prompt2全部拿来做training，prompt1全部拿来做testing。
            3. prompt1/2选100篇左右做testing，剩下全部做training（但要保证training里面有分数是1的文章）。
                1. Trainging+val / testing = 38:10
                2. training / val = 9:1       

        Args:
            df (pd.DataFrame): input data
            train_on_form (int, optional): train only on form N. Default (None) means both .
        """
        integer_score_only = self.config.integer_score_only
        df_copy = df.copy()
        
        if train_on_form:
            train_frac = 0.9
            val_frac = -1
            test_frac = -1
        else:
            train_val_frac = 38 / (38 + 10)
            train_frac = 0.9 * train_val_frac
            val_frac = 0.1 * train_val_frac
            test_frac = -1
        
        df_train = self.sample_2(df_copy, frac=train_frac, filter_form=train_on_form, integer_score_only=integer_score_only)
        df_copy.drop(df_train.index, inplace=True)
        # For validation set: take all remaining samples in the train_on_form group
        df_val = self.sample_2(df_copy, frac=val_frac, filter_form=train_on_form, integer_score_only=integer_score_only)
        df_copy.drop(df_val.index, inplace=True)
        df_test = self.sample_2(df_copy, frac=test_frac) # take all remaining samples
        
        self.print_distribution(df_train, df_val, df_test)

        write_data(df_train, self.config.index_train_filename)
        write_data(df_test, self.config.index_test_filename)
        write_data(df_val, self.config.index_val_filename)
    
    def sample_2(self, df: pd.DataFrame, frac=-1, filter_form=None, integer_score_only=False) -> pd.DataFrame:
        if integer_score_only:
            df = df[df[self.config.column_score] % 1 == 0].copy()

        if filter_form:
            df = df[df[self.config.column_form] == filter_form]
        
        if frac < 0:
            return df
            
        def calc_n_samples(frac, n):
            val = math.ceil(n * frac)
            if filter_form:
                if val == 0:
                    val = n
                elif val >= 2 and n - val <= 0:
                    val = n - 1
            else:
                # HACK: special care to make the distribution more balanced
                if n == 3:
                    val = 1
                if n == 2:
                    val = 1
                if n == 5:
                    val = 3
            return val        

        sampled_data = pd.concat([group.sample(calc_n_samples(frac, len(group))) 
                                  for key, group in df.groupby(self.config.column_score)])
        return sampled_data

    def split_data_3(self, df, train_on_form=None):
        """Split data into train/val/test sets
            prompt1/2选400篇左右做training，剩下全部做val（但要保证training里面有分数是1的文章）。
              - Trainging / val = 40:8

        Args:
            df (pd.DataFrame): input data
            train_on_form (int, optional): train only on form N. Default (None) means both .
        """
        integer_score_only = self.config.integer_score_only
        df_copy = df.copy()
        
        if train_on_form:
            raise ValueError("train_on_form is not supported for split_algorithm 3")
        else:
            train_frac = 400 / 480
        
        df_train = self.sample_3(df_copy, frac=train_frac, filter_form=train_on_form, integer_score_only=integer_score_only)
        df_copy.drop(df_train.index, inplace=True)
        # For validation set: take all remaining samples
        df_val = df_copy
        
        self.print_distribution(df_train, df_val)

        write_data(df_train, self.config.index_train_filename)
        write_data(df_val, self.config.index_val_filename)
        
    def sample_3(self, df: pd.DataFrame, frac=-1, filter_form=None, integer_score_only=False) -> pd.DataFrame:
        if integer_score_only:
            df = df[df[self.config.column_score] % 1 == 0].copy()

        if filter_form:
            df = df[df[self.config.column_form] == filter_form]
        
        if frac < 0:
            return df
            
        def calc_n_samples(frac, n):
            val = math.ceil(n * frac)
            if filter_form:
                if val == 0:
                    val = n
                elif val >= 2 and n - val <= 0:
                    val = n - 1
            else:
                # HACK: special care to make the distribution more balanced
                if n == 3:
                    val = 1
                if n == 2:
                    val = 1
                if n == 5:
                    val = 3
            return val        

        sampled_data = pd.concat([group.sample(calc_n_samples(frac, len(group))) 
                                  for key, group in df.groupby(self.config.column_score)])
        return sampled_data
    
    def print_distribution(self, df_train, df_val, df_test=None):
        # Get the unique score values across all DataFrames
        dfs = [df_train, df_val]
        if df_test is not None:
            dfs.append(df_test)
        all_scores = pd.concat([df[self.config.column_score] for df in dfs]).unique()
        all_scores = sorted(all_scores)

        # Create a DataFrame to hold the counts
        counts = pd.DataFrame({'SCORE': all_scores})

        # Count occurrences in each DataFrame
        counts['Train'] = counts['SCORE'].map(df_train[self.config.column_score].value_counts()).fillna(0).astype(int)
        counts['Val'] = counts['SCORE'].map(df_val[self.config.column_score].value_counts()).fillna(0).astype(int)
        if df_test is not None:
            counts['Test'] = counts['SCORE'].map(df_test[self.config.column_score].value_counts()).fillna(0).astype(int)

        # Add a sum row
        if df_test is not None:
            sum_row = counts[['Train', 'Val', 'Test']].sum().to_frame().T
        else:
            sum_row = counts[['Train', 'Val']].sum().to_frame().T
        sum_row.insert(0, 'SCORE', 'Sum')  # Insert a 'Sum' label in the SCORE column
        counts = pd.concat([counts, sum_row], ignore_index=True)  # Concatenate the sum row to the DataFrame
        
        print(counts)
        write_data(counts, self.config.dataset_distribution_filename, index=False)


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
        # Test data for openai fine-tuned model
        self.prepare(
            input_file=self.config.index_test_filename,
            system_message=self.config.system_message_short,
            dataset_fn=self.config.dataset_test_short_filename,
            skip_if_exist=skip_if_exist,
            for_training=False,
        )
        
        for dp in self.config.data_paths:
            if dp.is_finetuned:
                continue
            # Test data for baseline models
            self.prepare(
                input_file=self.config.index_test_filename,
                system_message=self.config.system_message_full,
                dataset_fn=dp.dataset_in,
                skip_if_exist=skip_if_exist,
                for_training=False,
                format=dp.format,
            )
        # # Test data for openai official models
        # self.prepare(
        #     input_file=self.config.index_test_filename,
        #     system_message=self.config.system_message_full,
        #     dataset_fn=self.config.dataset_test_full_filename,
        #     skip_if_exist=skip_if_exist,
        #     for_training=False,
        # )
        # # Test data for google official models
        # self.prepare(
        #     input_file=self.config.index_test_filename,
        #     system_message=self.config.system_message_full,
        #     dataset_fn=self.config.get_dataset_test_input_filename(model_id='gemini-pro'),
        #     skip_if_exist=skip_if_exist,
        #     for_training=False,
        #     format='gemini',
        # )

    def prepare(self, input_file, system_message, dataset_fn, chunk_size=1, skip_if_exist=True, for_training=True, format='openai'):
        if skip_if_exist and os.path.exists(dataset_fn):
            logger.debug(f"Dataset {dataset_fn} already exists, skip.")
            return
        if not os.path.exists(input_file):
            logger.warning(f"Input file {input_file} does not exist.")
            return
        essay_list = Essay.load_essays_from_file(input_file)
        dataset = []
        for chunk in self.divide_chunks(essay_list, chunk_size):
            if format == 'openai':
                record = self.convert_essay_to_dataset_openai(chunk, system_message, for_training)
            elif format == 'gemini':
                record = self.convert_essay_to_dataset_gemini(chunk, system_message)
            dataset.append(record)
        save_to_jsonl(dataset, dataset_fn)

        # Initial dataset stats
        logger.info(f"Num of records: [{len(dataset)}] in {dataset_fn}")
        # format_errors = self.format_check(dataset, for_training=for_training)
        # if format_errors:
        #     logger.error(f"Found errors: {format_errors}")
        #     return

        # print_stats(dataset)

    @classmethod
    def convert_essay_to_dataset_openai(cls, essays: list[Essay], system_message=None, for_training=True) -> dict:
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

    @classmethod
    def convert_essay_to_dataset_gemini(cls, essays: list[Essay], system_message=None) -> dict:
        # Initializing the messages list
        messages = []

        # Iterating through the lines and formatting the messages
        for essay in essays:
            # Formatting the message
            message = {
                "parts": [{"text": cls.prompt_formatter_full(essay, system_message=system_message)}]
            }
            messages.append(message)
            
        # Creating the final output dictionary
        output_dict = {
            "contents": messages,
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
            if not dp.active:
                continue
            self.parse_response(
                input_file=dp.dataset_out,
                output_file=dp.result_file,
                integer_score_only=self.config.integer_score_only,
                format=dp.format,
                skip_if_exist=skip_if_exist,
            )
            
    def parse_response(self, input_file, output_file, integer_score_only, format='openai', skip_if_exist=True):
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
                llm_prompt, raw_response = self.extract_prompt_and_raw_response(json_data, format)
                res = self.parse_raw_response(raw_response)
                essay_data = json_data[2]["essay"]
                agreement = calc_agreement(
                    ground_truth_score=essay_data["ETS Score"],
                    llm_score=res["score"],
                    integer_score_only=integer_score_only,
                    )
                row_data = {
                    **agreement,
                    "LLM Score": res["score"],
                    **essay_data,
                    "llm_prompt": llm_prompt,
                    "raw_response": raw_response,
                    "reasoning": res["reasoning"],
                }
                rows.append(row_data)
        df = pd.DataFrame(rows)
        write_data(df, output_file)
        logger.info(f"Parsed result saved to file: {output_file}")
    
    @classmethod
    def extract_prompt_and_raw_response(cls, json_data, format):
        raw_response = ''
        llm_prompt = ''
        if format == 'openai':
            for message in json_data[0]["messages"]:
                if message["role"] == "user":
                    llm_prompt = message["content"]
                    break
            raw_response = json_data[1]["choices"][0]["message"]["content"]
        elif format == 'gemini':
            llm_prompt = json_data[0]["contents"][0]["parts"][0]["text"]
            raw_response = json_data[1]["candidates"][0]["content"]["parts"][0]["text"]
        return llm_prompt, raw_response
    
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
        if skip_if_exist and os.path.exists(self.config.result_summary_filename):
            print("Result summary already exists, skip.")
            return
    
        logger.info(f"Generating result summary: {self.config.result_summary_filename}")
        
        def combine_labels(rate_label, model_label):
            return f"{model_label}\n{rate_label}"
        res = {
            'label': self.config.run_prefix,
            'train_on_form': self.config.tof_name,
            'integer_score_only': self.config.integer_score_only,
            'train_size': self.calc_dataset_size(self.config.index_train_filename),
            'val_size': self.calc_dataset_size(self.config.index_val_filename),
            'test_size': self.calc_dataset_size(self.config.index_test_filename),
            'test_size_p1': self.calc_dataset_size(self.config.index_test_filename, filter_form=1),
            'test_size_p2': self.calc_dataset_size(self.config.index_test_filename, filter_form=2),
        }
        model_labels = [dp.llm_model_label for dp in self.config.data_paths]
        rate_labels = []
        tmp = {}
        for dp in self.config.data_paths:
            rate_dict = calc_metrics_dict(dp.result_file, self.config.integer_score_only)
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
        
    def calc_dataset_size(self, fn, filter_form=None):
        df = read_data(fn)
        if filter_form:
            df = df[df[self.config.column_form] == filter_form]
        return df.shape[0]