import os
import datetime

uploaded_file_id_fn = 'file-id-eassy-test-1-2023-10-22-19-23-05.json'


date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
model_suffix = 'eassy-test-1'

input_root= 'data/input/AWE'
index_path = os.path.join(input_root, 'index', 'index.csv')

essay_root = os.path.join(input_root, 'responses', 'original')

output_root = 'data/output'

index_train_filename = os.path.join(output_root, 'index', 'train.csv')
index_val_filename = os.path.join(output_root, 'index', 'val.csv')
index_test_filename = os.path.join(output_root, 'index', 'test.csv')

dataset_train_filename = os.path.join(output_root, 'dataset', 'train.jsonl')
dataset_val_filename = os.path.join(output_root, 'dataset', 'val.jsonl')
dataset_test_filename = os.path.join(output_root, 'dataset', 'test.jsonl')

file_id_filename = os.path.join(output_root, 'ids', f'file-id-{model_suffix}-{date_str}.json')
model_id_filename = os.path.join(output_root, 'ids', f'model-id-{model_suffix}-{date_str}.json')
job_id_filename = os.path.join(output_root, 'ids', f'job-id-{model_suffix}-{date_str}.json')
uploaded_file_id_filename = os.path.join(output_root, 'ids', uploaded_file_id_fn)

num_per_group = {
    'train': 30,
    'val': 10,
    'test': 10
}

num_per_group = {
    'train': 10,
    'val': 2,
    'test': 10
}

num_of_essays_per_prompt = 1


system_message = """You are a language expert who evaluate user's input with 3 levels (low/medium/high), based on the following rubrics:
An essay with a "high" level accomplishes at least the following:
* Addresses the topic and task well, though some points may not be fully elaborated.
* Is generally well organized and well developed, using clearly appropriate and sufficient explanations, exemplifications, and/or details.
* Displays unity, progression, and coherence., though there may be occasional redundancy, digression, or unclear connections.
* Demonstrates facility in the use of language, showcasing syntactic variety, appropriate word choice in a good range of vocabulary, and idiomaticity. While minor lexical or grammatical errors might be present, they do not interfere with meaning.

An essay with a "medium" level is marked by one or more of the following:
* Generally addresses the topic and task.
* Offers somewhat developed explanations, exemplifications, and/or details, though there might be moments where elaboration is lacking or slightly off-target.
* Displays unity, progression, and coherence for the most part, but the connection of ideas may be occasionally obscured or lack smooth transitions.
* Demonstrates the ability to use certain syntactic structures and vocabulary, but the range of these structures and vocabulary is limited. There might be inconsistencies in sentence formation and word choice leading to occasional lack of clarity or slightly obscured meaning.

An essay with a "low" level may reveal one or more of the following weaknesses:
* Limited development in response to the topic and task or serious disorganization or underdevelopment.
* Inadequate organization or connection of ideas, with little or no detail, irrelevant specifics, or questionable responsiveness to the task.
* Inappropriate or insufficient exemplifications, explanations, or details to support or illustrate generalizations.
* A noticeably inappropriate choice of words, word forms, or serious and frequent errors in sentence structure or usage."""
