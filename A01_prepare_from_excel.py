import logging
import os
import pandas as pd
from lib.utils import setup_log
from lib.data_processing import DatasetPreparation
from lib.config import MyConfig
from lib.essay import Essay
from lib.io import save_to_jsonl
from configlist import config_list
import setting

logger = logging.getLogger(__name__)

skip_if_exist = True
skip_if_exist = False


input_file = "./data/output/2025-02-17-inference-on-new-test-set-ToF[all]-IntOnly[False]/index/testdataset1_all_2.17.2025.xlsx"

class SpecialDatasetPreparation(DatasetPreparation):
    def __init__(self, config) -> None:
        super().__init__(config)

    def run(self, skip_if_exist=True):
        # Test data for openai fine-tuned model
        self.prepare(
            input_file=input_file,
            system_message=self.config.system_message_short,
            dataset_fn=self.config.dataset_test_short_filename,
            skip_if_exist=skip_if_exist,
            for_training=False,
        )
    
    def prepare(self, input_file, system_message, dataset_fn, chunk_size=1, skip_if_exist=True, for_training=True, format='openai'):
        if skip_if_exist and os.path.exists(dataset_fn):
            logger.debug(f"Dataset {dataset_fn} already exists, skip.")
            return
        if not os.path.exists(input_file):
            logger.warning(f"Input file {input_file} does not exist.")
            return
        essay_list = self.load_essays_from_excel(input_file)
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
        
    def load_essays_from_excel(self, input_file):
        df = pd.read_excel(input_file)
        def create_object(row):
            obj = Essay(
                fn="",
                text=row["essay"],
                score=row["ETS Score"],
                prompt_text=row["essay_prompt"],
            )
            return obj

        objs = df.apply(create_object, axis=1).tolist()
        return objs


def main():
    for config_files in config_list:
        config = MyConfig(file_paths=config_files)

        prepare = SpecialDatasetPreparation(config)
        prepare.run(skip_if_exist=skip_if_exist)

    
if __name__ == "__main__":
    setup_log()
    main()
