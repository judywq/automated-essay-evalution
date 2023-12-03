import os
import json
import datetime
from typing import Optional
from pydantic import BaseModel


class DataPath(BaseModel):
    index_file: str
    dataset_in: str
    dataset_out: str
    result_file: str
    is_official: bool = False
    llm_model_label: Optional[str] = ""
    llm_model_name: Optional[str] = None


class JsonConfigLoader:
    def __init__(self, file_paths=[]):
        self.data = {}
        self.load_configs(file_paths)

    def load_configs(self, file_paths: list[str] | str):
        if not file_paths:
            return
        if isinstance(file_paths, str):
            if file_paths:
                file_paths = [file_paths]
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as file:
                    config = json.load(file)
                    self.data.update(config)
            except FileNotFoundError:
                print(f"Warning: Configuration file not found at {file_path}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in the configuration file at {file_path}")

    def __getattr__(self, key):
        return self.data[key]

class MyConfig(JsonConfigLoader):
    
    def __init__(self, file_paths=[]):
        super().__init__(file_paths)
        self.date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.system_message_map = {}
        self.data_paths = []
        
    def load_configs(self, file_paths: list[str] | str):
        if not file_paths:
            return
        super().load_configs(file_paths)
        self.load_data_paths()
    
    def load_data_paths(self):
        try:
            data_paths = []
            for model_name in self.baseline_models:
                data_paths.append(DataPath(
                    index_file=self.index_test_filename,
                    dataset_in=os.path.join(self.output_root, 'dataset', 'test.full.jsonl'),
                    dataset_out=os.path.join(self.output_root, 'dataset', f'test.full.result.{model_name}.jsonl'),
                    result_file=os.path.join(self.output_root, 'results', f'test-result-{model_name}.xlsx'),
                    llm_model_label=model_name,
                    llm_model_name=model_name,
                    is_official=True,   
                ))
            data_paths.append(DataPath(
                index_file=self.index_test_filename,
                dataset_in=os.path.join(self.output_root, 'dataset', 'test.short.jsonl'),
                dataset_out=os.path.join(self.output_root, 'dataset', f'test.short.result.{self.test_result_prefix}.jsonl'),
                result_file=os.path.join(self.output_root, 'results', f'test-result-{self.test_result_prefix}.xlsx'),
                llm_model_label="finetuned-gpt-3.5",
                is_official=False,
            ))
            self.data_paths = data_paths
        except KeyError:
            # Probably the config file is not provided yet
            pass
            
    @property
    def tof_name(self):
        return self.train_on_form if self.train_on_form else 'all'
    
    @property
    def global_run_id(self):
        return f'{self.run_prefix}-ToF[{self.tof_name}]-IntOnly[{self.integer_score_only}]'
    
    @property
    def output_root(self):
        return os.path.join('data/output', self.global_run_id)

    @property
    def index_train_filename(self):
        return os.path.join(self.output_root, 'index', 'train.csv')
    
    @property
    def index_val_filename(self):
        return os.path.join(self.output_root, 'index', 'val.csv')
    
    @property
    def index_test_filename(self):
        return os.path.join(self.output_root, 'index', 'test.csv')
    
    @property
    def dataset_train_filename(self):
        return os.path.join(self.output_root, 'dataset', 'train.jsonl')
    
    @property
    def dataset_val_filename(self):
        return os.path.join(self.output_root, 'dataset', 'val.jsonl')
    
    @property
    def dataset_test_short_filename(self):
        return os.path.join(self.output_root, 'dataset', 'test.short.jsonl')
    
    @property
    def dataset_test_full_filename(self):
        return os.path.join(self.output_root, 'dataset', 'test.full.jsonl')
    
    def get_dataset_test_result_filename(self, input_fn, model_name):
        return input_fn.replace('.jsonl', f'.result.{model_name}.jsonl')
    
    def get_dataset_test_result_finetuned_filename(self):
        input_fn = self.dataset_test_short_filename
        return input_fn.replace('.jsonl', f'.result.{self.test_result_prefix}.jsonl')
    
    @property
    def file_id_filename(self):
        return os.path.join(self.output_root, 'ids', f'file-id.json')
    
    @property
    def job_id_filename(self):
        return os.path.join(self.output_root, 'ids', f'job-id.json')
    
    @property
    def test_result_finetuned_filename(self):
        return os.path.join(self.output_root, 'results', f'test-result-{self.test_result_prefix}-{self.date_str}.xlsx')
    
    @property
    def test_result_official_filename(self):
        # TODO: rename DEFAULT_MODEL to inference_model
        return os.path.join(self.output_root, 'results', f'test-result-{self.DEFAULT_MODEL}-{self.date_str}.xlsx')
    
    @property
    def result_summary_filename(self):
        return os.path.join(self.output_root, f'result-summary.xlsx')
    
    @property
    def extra_limitation(self):
        return "" if self.integer_score_only else "(with 0.5 increments)"
    
    @property
    def system_message_short(self):
        return self.load_system_message(self.system_message_short_fn)

    @property
    def system_message_full(self):
        return self.load_system_message(self.system_message_full_fn)
    
    def load_system_message(self, fn):
        cache = self.system_message_map.get(fn, None)
        if cache:
            return cache
        try:
            with open(fn, 'r') as file:
                txt = file.read().strip().format(extra_limitation=self.extra_limitation)
                self.system_message_map[fn] = txt
                return txt
        except FileNotFoundError:
            raise FileNotFoundError(f"System message file not found at {fn}")
