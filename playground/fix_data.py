import os
import json
import shutil
from icecream import ic
from lib.data_processing import DataSplitter
from lib.essay import Essay
from lib.utils import setup_log
from lib.config import MyConfig
from lib.io import save_to_jsonl
from configlist import *


def append_metadata():
    for config_files in [
        config_set_fa_float,
        # config_set_f1_float, # Done
        config_set_f2_float,
        config_set_fa_int,
        # config_set_f1_int, # Done
        config_set_f2_int,
    ]:
        config = MyConfig()
        config.load_configs(file_paths=config_files)
        
        splitter = DataSplitter(config)
        df = splitter.read_all_data()
        essays = Essay.load_essays_from_dataframe(df)
        
        mapping = { essay.text: essay for essay in essays }
        
        for input_fn in [
            config.dataset_test_short_filename,
            config.dataset_test_full_filename,
        ]:
            input_fn_bk = input_fn.replace(".jsonl", ".bk.jsonl")
            
            # if os.path.exists(input_fn_bk):
                # ic("Skip", input_fn)
                # continue
            
            new_data = []
            with open(input_fn, "r") as f:                
                for line in f:
                    request_json = json.loads(line)
                    # ic(request_json)
                    prompt = request_json["messages"][0]["content"]
                    for essay_text, essay in mapping.items():
                        if essay_text in prompt:
                            request_json["metadata"] = { "essay": essay.to_dict() }
                            new_data.append(request_json)
                            break
            # backup file first
            shutil.copy(input_fn, input_fn_bk)
            save_to_jsonl(new_data, input_fn)
        
        for baseline_model in config.baseline_models:
            input_fn = config.dataset_test_full_filename
            output_fn = config.get_dataset_test_result_filename(
                input_fn=input_fn,
                model_name=baseline_model,
            )
            
            output_fn_bk = output_fn.replace(".jsonl", ".bk.jsonl")
            
            if os.path.exists(output_fn_bk):
                continue
            
            new_data = []
            with open(output_fn, "r") as f:
                for line in f:
                    request_json = json.loads(line)
                    # ic(request_json)
                    prompt = request_json[0]["messages"][0]["content"]
                    for essay_text, essay in mapping.items():
                        if essay_text in prompt:
                            request_json.append({
                                "essay": essay.to_dict(),
                            })
                            new_data.append(request_json)
                            break
                    # break
            # backup file first
            shutil.copy(output_fn, output_fn_bk)
            # save_to_jsonl(new_data, output_fn.replace(".jsonl", ".with_metadata.jsonl"))
            save_to_jsonl(new_data, output_fn)
            # print(output_fn)
            # break
        # break
    ic("Done")
    

if __name__ == "__main__":
    setup_log()
    append_metadata()
