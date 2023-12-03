from lib.utils import setup_log
from lib.data_processing import DataSplitter, DatasetPreparation
from lib.config import MyConfig
from configlist import config_list

skip_if_exist = True
# skip_if_exist = False

def main():
    for config_files in config_list:
        config = MyConfig(file_paths=config_files)

        splitter = DataSplitter(config)
        splitter.run(skip_if_exist=skip_if_exist)

        prepare = DatasetPreparation(config)
        prepare.run(skip_if_exist=skip_if_exist)

    
if __name__ == "__main__":
    setup_log()
    main()
