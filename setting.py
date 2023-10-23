import os
import datetime

global_run_id = '2023-10-23-update-rubrics'
model_suffix = 'eassy-test-4'

# date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

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
test_result_filename = os.path.join(output_root, 'results', f'test-result.xlsx')

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

system_message = """You are a language expert who evaluates test-takers' argumentative essays on 3 levels (low/medium/high), based on the given essay prompt and rubrics. The prompt will be given together with the essay later, and the rubrics are as follows:
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
