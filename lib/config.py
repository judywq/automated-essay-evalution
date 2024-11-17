import os
import json
import datetime
from typing import Optional
from pydantic import BaseModel


class LlmModel(BaseModel):
    active: bool
    id: str
    label: str
    format: str

class DataPath(BaseModel):
    index_file: str
    dataset_in: str
    dataset_out: str
    result_file: str
    is_finetuned: bool = False
    llm_model_label: Optional[str] = ""
    llm_model_id: Optional[str] = None
    active: bool = True
    format: str = "openai"


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
        self.data_paths = []
        self.date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.system_message_map = {}
        super().__init__(file_paths)

    def load_configs(self, file_paths: list[str] | str):
        if not file_paths:
            return
        super().load_configs(file_paths)
        self.load_data_paths()
    
    def load_data_paths(self):
        try:
            data_paths = []
            if self.run_baseline:
                for model in self.get_baseline_models():
                    model_id = model.id
                    model_label = model.label
                    data_paths.append(DataPath(
                        index_file=self.index_test_filename,
                        dataset_in=self.get_dataset_test_input_filename(model_id),
                        dataset_out=self.get_dataset_test_output_filename(model_id),
                        result_file=self.get_test_result_filename(model_id),
                        llm_model_label=model_label,
                        llm_model_id=model_id,
                        format=model.format,
                        is_finetuned=False,
                        active=model.active,
                    ))
            # Finetuned model
            data_paths.append(DataPath(
                index_file=self.index_test_filename,
                dataset_in=os.path.join(self.output_root, 'dataset', 'test.short.jsonl'),
                dataset_out=os.path.join(self.output_root, 'dataset', f'test.short.result.{self.finetuned_prefix}.jsonl'),
                result_file=os.path.join(self.output_root, 'results', f'test-result-{self.finetuned_prefix}.xlsx'),
                # llm_model_label="finetuned-gpt-3.5",
                llm_model_label=self.finetuned_prefix,
                is_finetuned=True,
                active=self.run_finetuned,
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
    def distribution_filename(self):
        return os.path.join(self.output_root, 'index', 'score_distribution.csv')
    
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

    def get_dataset_test_input_filename(self, model_id:str):
        # if model_id not in ['gemini-pro']:
            # raise ValueError(f"Unsupported model id: {model_id}")
        return os.path.join(self.output_root, 'dataset', f'test.full.{model_id}.jsonl')
    
    def get_dataset_test_output_filename(self, model_id):
        return os.path.join(self.output_root, 'dataset', f'test.full.{model_id}.output.jsonl')
        # return input_fn.replace('.jsonl', f'.result.{model_id}.jsonl')
    
    def get_dataset_test_result_finetuned_filename(self):
        input_fn = self.dataset_test_short_filename
        return input_fn.replace('.jsonl', f'.result.{self.finetuned_prefix}.jsonl')
    
    @property
    def file_id_filename(self):
        return os.path.join(self.output_root, 'ids', f'file-id.json')
    
    @property
    def job_id_filename(self):
        return os.path.join(self.output_root, 'ids', f'job-id.json')
    
    @property
    def test_result_finetuned_filename(self):
        return os.path.join(self.output_root, 'results', f'test-result-{self.finetuned_prefix}-{self.date_str}.xlsx')
    
    def get_test_result_filename(self, model_id):
        return os.path.join(self.output_root, 'results', f'test-result-{model_id}.xlsx')
    
    @property
    def result_summary_filename(self):
        return os.path.join(self.output_root, f'result-summary.xlsx')

    @property
    def total_result_summary_filename(self):
        return os.path.join(self.output_root, f'total-result-summary.xlsx')
    
    def get_baseline_models(self) -> list[LlmModel]:
        results = []
        # for model_id in self.baseline_models:
        for model in self.llm_models:
            active = model['id'] in self.baseline_models
            results.append(LlmModel(**model, active=active))
        return results
    
    def get_llm_model(self, model_id: str) -> LlmModel:
        for model in self.llm_models:
            if model['id'] == model_id:
                return LlmModel(**model)
        raise ValueError(f"LLM model not found: {model_id}")
    
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
