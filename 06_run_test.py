import os
import json
import re
from time import sleep
import openai
from icecream import ic
import pandas as pd
from lib.essay import Essay
from lib.io import write_data
from lib.utils import prompt_formatter
import setting


def main():
    # system_message = setting.system_message
    system_message = setting.system_message_short
    limit_num = 300
    
    resp = json.load(open(setting.job_id_filename, "r"))
    fine_tuned_model_id = resp["fine_tuned_model"]
    print(fine_tuned_model_id)

    essays = Essay.load_essays(
        index_file=setting.index_test_filename, 
        essay_root=setting.essay_root,
        prompt_root=setting.essay_prompt_root,
    )

    result = []
    for index, essay in enumerate(essays):
        result.append(run_model(fine_tuned_model_id, essay, system_message))

        if index + 1 >= limit_num:
            break
        
        if index % 10 == 0:
            df = pd.DataFrame(result)
            write_data(df, setting.test_result_filename)
            
        if index % 60 == 0:
            sleep(10)

    df = pd.DataFrame(result)
    write_data(df, setting.test_result_filename)
    
    success_rate = df["ok"].sum() / len(df)
    print("Done! Success rate:", success_rate)


def run_model(fine_tuned_model_id, essay, system_message):
    # Generating using the new model
    test_messages = []
    test_messages.append({"role": "system", "content": system_message})
    test_messages.append({"role": "user", "content": prompt_formatter(essay)})
    # ic(test_messages)

    response = openai.ChatCompletion.create(
        model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=500
    )
    raw_response = response["choices"][0]["message"]["content"]
    # ic(raw_response)

    raw_response = raw_response.lower()
    pat = re.compile(f"(high|medium|low|none)")
    match = pat.search(raw_response)
    if match:
        level_resp = match.group(0)
    else:
        level_resp = ""

    ok = level_resp == essay.level.value
    tag = "OK" if ok else "NG"
    print(f"{tag} - {essay.level.value} vs {level_resp}")

    res = {
        "ok": ok,
        "level": essay.level.value,
        "leve_resp": level_resp,
        "filename": essay.fn,
        "prompt": essay.prompt,
        "raw_response": raw_response,
        "text": essay.text,
    }
    return res


if __name__ == "__main__":
    main()
