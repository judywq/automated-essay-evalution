import pandas as pd
from lib.io import write_data
from lib.utils import setup_log
from lib.data_processing import SummaryGenerator
from lib.config import MyConfig
from configlist import config_list

skip_if_exist = True
skip_if_exist = False

def main():
    all_data = []
    for config_files in config_list:
        config = MyConfig(file_paths=config_files)
        
        total_summary_fn = config.total_result_summary_filename
        
        gen = SummaryGenerator(config=config)
        res = gen.run(skip_if_exist=skip_if_exist)
        all_data.append(res)
        # break

    write_data(pd.DataFrame(all_data), total_summary_fn)


if __name__ == "__main__":
    setup_log()
    main()
