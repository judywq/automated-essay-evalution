import os
import datetime

train_on_form = 1
# train_on_form = 2
# train_on_form = None

tof_name = train_on_form if train_on_form else 'all'

# integer_score_only = False
integer_score_only = True

global_run_id = f'2023-12-02-fixed-data-length-ToF[{tof_name}]-IntOnly[{integer_score_only}]'
model_suffix = 'eassy-all'
test_result_prefix = 'finetuned-gpt-3.5'


DEFAULT_LOG_LEVEL = "INFO"
# DEFAULT_LOG_LEVEL = "DEBUG"

official_model_gpt_3_5_turbo = 'gpt-3.5-turbo'
official_model_gpt_4 = 'gpt-4'
official_model_gpt_4_turbo = 'gpt-4-1106-preview'
DEFAULT_MODEL = official_model_gpt_3_5_turbo
# DEFAULT_MODEL = official_model_gpt_4_turbo

REQUEST_TIMEOUT_SECS = 60


date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

index_file_path_template = './data/input/TOEFL-iBT/Form {form_id:d}/writing_Form{form_id:d}.xlsx'
response_file_path_tempalte = './data/input/TOEFL-iBT/Form {form_id:d}/Writing Responses - Form {form_id:d}/Form{form_id:d}-{response_id:d}-Item{item_id:d}.txt'
prompt_file_path_tempalte = './data/input/TOEFL-iBT/Form {form_id:d}/Prompts/prompt_item_{item_id:d}.txt'

output_root = os.path.join('data/output', global_run_id)

index_train_filename = os.path.join(output_root, 'index', 'train.csv')
index_val_filename = os.path.join(output_root, 'index', 'val.csv')
index_test_filename = os.path.join(output_root, 'index', 'test.csv')

dataset_train_filename = os.path.join(output_root, 'dataset', 'train.jsonl')
dataset_val_filename = os.path.join(output_root, 'dataset', 'val.jsonl')
dataset_test_filename = os.path.join(output_root, 'dataset', 'test.jsonl')

file_id_filename = os.path.join(output_root, 'ids', f'file-id.json')
job_id_filename = os.path.join(output_root, 'ids', f'job-id.json')
test_result_filename = os.path.join(output_root, 'results', f'test-result-{test_result_prefix}-{date_str}.xlsx')
test_result_official_filename = os.path.join(output_root, 'results', f'test-result-{DEFAULT_MODEL}-{date_str}.xlsx')

fine_tuning_base_model_id = 'gpt-3.5-turbo-1106'
# base_model_id = 'ft:gpt-3.5-turbo-0613:waseda-university:eassy-test-2:8CTA9Ik1'

# The column names in the input data
column_response_id = 'Sample_ID' # response id
column_score = 'Independent' # essay score
column_form = 'Form_ID' # form id
column_item = 'Item_ID' # item id


# item id, 1 is for Integrated, 2 is for Independent
item_id = 2

num_per_group = {
    'train': 10,
    'val': 3,
    'test': -1
}

# num_per_group = {
#     'train': 10,
#     'val': 2,
#     'test': 10
# }

num_of_essays_per_prompt = 1

extra_limitation = "" if integer_score_only else "(with 0.5 increments)"

system_message_short = f"""As a language expert, your task is to evaluate argumentative essays on a scale of 0 to 5 {extra_limitation}, based on the pre-trained rubrics."""

system_message_score = f"""As a language expert, your task is to evaluate argumentative essays on a scale of 0 to 5 {extra_limitation}\
based on the rubrics below.

5 points:
- Effectively addresses the topic and task 
- Is well organized and well developed, using clearly appropriate explanations, exemplifications and/or details 
- Displays unity, progression and coherence 
- Displays consistent facility in the use of language, demonstrating syntactic variety, appropriate word choice and idiomaticity, though it may have minor lexical or grammatical errors 

4 points:
- Addresses the topic and task well, though some points may not be fully elaborated 
- Is generally well organized and well developed, using appropriate and sufficient explanations, exemplifications and/or details 
- Displays unity, progression and coherence, though it may contain occasional redundancy, digression, or unclear connections 
- Displays facility in the use of language, demonstrating syntactic variety and range of vocabulary, though it will probably have occasional noticeable minor errors in structure, word form or use of idiomatic language that do not interfere with meaning 

3 points:
- Addresses the topic and task using somewhat developed explanations, exemplifications and/or details 
- Displays unity, progression and coherence, though connection of ideas may be occasionally obscured 
- May demonstrate inconsistent facility in sentence formation and word choice that may result in lack of clarity and occasionally obscure meaning 
- May display accurate but limited range of syntactic structures and vocabulary 

2 points:
- Limited development in response to the topic and task 
- Inadequate organization or connection of ideas 
- Inappropriate or insufficient exemplifications, explanations or details to support or illustrate generalizations in response to the task 
- A noticeably inappropriate choice of words or word forms 
- An accumulation of errors in sentence structure and/or usage 

1 point:
- Serious disorganization or underdevelopment 
- Little or no detail, or irrelevant specifics, or questionable responsiveness to the task 
- Serious and frequent errors in sentence structure or usage

0 point:
- An essay at this level merely copies words from the topic, rejects the topic, or is otherwise not connected to the topic, is written in a foreign language, consists of keystroke characters, or is blank.
"""

system_message = system_message_score

# GPT