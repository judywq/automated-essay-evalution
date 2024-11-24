# Extract the text from the essays and save them to individual files

import os
import pandas as pd
from configlist import config_unit_base
from lib.config import MyConfig
import shutil
import re
import json

fn_index_file = './data/output/2024-11-19-gpt4o-new-split-ToF[all]-IntOnly[False]/index/test.csv'

output_dir = './data/output/2024-11-19-gpt4o-new-split-ToF[all]-IntOnly[False]/remaining-test-responses/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

new_txt_file_temp = 'Form[{form_id}]_Item[{item_id}]_Sample[{response_id}]_Score[{score}].txt'
new_json_file_temp = 'Form[{form_id}]_Item[{item_id}]_Sample[{response_id}]_Score[{score}].json'

def main():
    
    config = MyConfig(file_paths=[config_unit_base])
    df = pd.read_csv(fn_index_file)
    for i, row in df.iterrows():
        response_id = row['Sample_ID']
        item_id = row['Item_ID']
        score = row['Independent']
        form_id = row['Form_ID']
        fn_response = config.response_file_path_tempalte.format(response_id=response_id, form_id=form_id, item_id=item_id)
        fn_new_response_txt = os.path.join(output_dir, new_txt_file_temp.format(response_id=response_id, form_id=form_id, item_id=item_id, score=score))
        fn_new_response_json = os.path.join(output_dir, new_json_file_temp.format(response_id=response_id, form_id=form_id, item_id=item_id, score=score))
        
        # copy the txt as is
        shutil.copyfile(fn_response, fn_new_response_txt)
        
        # convert the txt to json
        with open(fn_response, 'r') as f:
            text = f.read()

        json_obj = text_to_json(text)
        num_of_paragraphs = len(json_obj['paragraphs'])
        num_of_sentences = sum([len(para['sentences']) for para in json_obj['paragraphs']])
        print(f"Form ID: {form_id}, Item ID: {item_id}, Response ID: {response_id}, #Paragraphs: {num_of_paragraphs}, #Sentences: {num_of_sentences}")

        with open(fn_new_response_json, 'w') as f:
            json.dump(json_obj, f, indent=4)

        if i > 10:
            pass
            # break
        


def text_to_json(text):
    """
    Converts a plain text essay into a structured JSON format with paragraph and sentence IDs.
    
    Args:
        text (str): The input essay as a plain text string.
        
    Returns:
        dict: The JSON representation of the essay.
    """
    text = text.strip()
    
    if '\n\n' in text:
        # Split the text into paragraphs using double line breaks
        paragraphs = text.split('\n\n')
    elif '\n' in text:
        # Split the text into paragraphs using single line breaks
        paragraphs = text.split('\n')
    elif '   ' in text:
        # Split the text into paragraphs using multiple spaces
        paragraphs = re.split(r'\s{5,}', text)
    else:
        # Assume the text is a single paragraph
        paragraphs = [text]
    
    # Initialize the JSON structure
    essay_json = {"paragraphs": []}
    
    # Process each paragraph
    for para_id, paragraph in enumerate(paragraphs, start=1):
        # Split sentences using a regex to detect sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
        
        # Add each sentence with an ID to the paragraph
        para_data = {
            "id": para_id,
            "sentences": [
                {"id": f"{para_id}.{sent_id}", "text": sentence.strip()}
                for sent_id, sentence in enumerate(sentences, start=1)
            ]
        }
        
        # Append the paragraph to the JSON
        essay_json["paragraphs"].append(para_data)
    
    return essay_json

if __name__ == '__main__':
    main()
