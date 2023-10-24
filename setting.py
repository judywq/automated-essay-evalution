import os
import datetime

DEFAULT_LOG_LEVEL = "INFO"
# DEFAULT_LOG_LEVEL = "DEBUG"

official_model_gpt_3_5_turbo = 'gpt-3.5-turbo'
official_model_gpt_4 = 'gpt-4'
DEFAULT_MODEL = official_model_gpt_4

REQUEST_TIMEOUT_SECS = 60

global_run_id = '2023-10-23-update-rubrics'
model_suffix = 'eassy-test-4'

date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

input_root= 'data/input/AWE'
index_path = os.path.join(input_root, 'index', 'index.csv')

essay_root = os.path.join(input_root, 'responses', 'original')
essay_prompt_root = os.path.join(input_root, 'prompts')

output_root = os.path.join('data/output', global_run_id)

index_train_filename = os.path.join(output_root, 'index', 'train.csv')
index_val_filename = os.path.join(output_root, 'index', 'val.csv')
index_test_filename = os.path.join(output_root, 'index', 'test.csv')

dataset_train_filename = os.path.join(output_root, 'dataset', 'train.jsonl')
dataset_val_filename = os.path.join(output_root, 'dataset', 'val.jsonl')
dataset_test_filename = os.path.join(output_root, 'dataset', 'test.jsonl')

file_id_filename = os.path.join(output_root, 'ids', f'file-id.json')
job_id_filename = os.path.join(output_root, 'ids', f'job-id.json')
test_result_filename = os.path.join(output_root, 'results', f'test-result-{date_str}.xlsx')

base_model_id = 'gpt-3.5-turbo'
# base_model_id = 'ft:gpt-3.5-turbo-0613:waseda-university:eassy-test-2:8CTA9Ik1'

num_per_group = {
    'train': 30,
    'val': 10,
    'test': 10
}

# num_per_group = {
#     'train': 10,
#     'val': 2,
#     'test': 10
# }

num_of_essays_per_prompt = 1

system_message_short = """You are a language expert who evaluates test-takers' argumentative essays on 3 levels (low/medium/high), based on the pre-trained rubrics."""

system_message_level = """You are a language expert who evaluates test-takers' argumentative essays on 3 levels (low/medium/high), based on the given essay prompt and rubrics. The prompt will be given together with the essay later, and the rubrics are as follows:
An essay at "high" level accomplishes the following:
* Effectively addresses the topic and task well.
* Is well organized and well developed, using clearly appropriate and sufficient explanations, exemplifications, and/or details.
* Displays unity, progression, and coherence.
* Demonstrates facility in the use of language, showcasing syntactic variety, appropriate word choice in a good range of vocabulary, and idiomaticity. While minor lexical or grammatical errors might be present, they do not interfere with meaning.

An essay at "medium" level accomplishes the following:
* Generally addresses the topic and task.
* Offers moderately developed explanations, exemplifications, and/or details, and there are moments where elaboration is lacking or slightly off-target.
* Displays unity, progression, and coherence for the most part, but the connection of ideas is occasionally obscured or lack smooth transitions.
* Demonstrates the ability to use certain syntactic structures and vocabulary, but the range of these structures and vocabulary is limited. There are occasional lexical or grammatical errors leading to lack of clarity or obscured meaning.

An essay at "low" level may reveal one or more of the following weaknesses:
* Underdevelopment in response to the topic and task or serious disorganization.
* Inadequate organization or connection of ideas, with little or no detail, irrelevant specifics, or questionable responsiveness to the task.
* Inappropriate or insufficient exemplifications, explanations, or details to support or illustrate generalizations.
* Many inappropriate choice of words, word forms, or serious and frequent errors in sentence structure or usage.
"""

system_message_score = """You are a language expert who evaluates test-takers' argumentative essays at a scale of 1 to 5, based on the given essay prompt and rubrics. The prompt will be given together with the essay later, and the rubrics are as follows:
An essay worth 5 points largely accomplishes all of the following: 
- Effectively addresses the topic and task 
- Is well organized and well developed, using clearly appropriate explanations, exemplifications and/or details 
- Displays unity, progression and coherence 
- Displays consistent facility in the use of language, demonstrating syntactic variety, appropriate word choice and idiomaticity, though it may have minor lexical or grammatical errors 
An essay worth 4 points largely accomplishes all of the following: 
- Addresses the topic and task well, though some points may not be fully elaborated 
- Is generally well organized and well developed, using appropriate and sufficient explanations, exemplifications and/or details 
- Displays unity, progression and coherence, though it may contain occasional redundancy, digression, or unclear connections 
- Displays facility in the use of language, demonstrating syntactic variety and range of vocabulary, though it will probably have occasional noticeable minor errors in structure, word form or use of idiomatic language that do not interfere with meaning 
An essay worth 3 points is marked by one or more of the following: 
- Addresses the topic and task using somewhat developed explanations, exemplifications and/or details 
- Displays unity, progression and coherence, though connection of ideas may be occasionally obscured 
- May demonstrate inconsistent facility in sentence formation and word choice that may result in lack of clarity and occasionally obscure meaning 
- May display accurate but limited range of syntactic structures and vocabulary 
An essay worth 2 points may reveal one or more of the following weaknesses: 
- Limited development in response to the topic and task 
- Inadequate organization or connection of ideas 
- Inappropriate or insufficient exemplifications, explanations or details to support or illustrate generalizations in response to the task 
- A noticeably inappropriate choice of words or word forms 
- An accumulation of errors in sentence structure and/or usage 
An essay worth 1 point is seriously flawed by one or more of the following weaknesses: 
- Serious disorganization or underdevelopment 
- Little or no detail, or irrelevant specifics, or questionable responsiveness to the task 
- Serious and frequent errors in sentence structure or usage 
"""

system_message = system_message_score

# GPT