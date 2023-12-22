from lib.utils import setup_log
from lib.model_runner import ModelRunner
from lib.data_processing import ResponseParser
from lib.config import MyConfig
from configlist import config_list

skip_if_exist = True
skip_if_exist = False

def main():
    for config_files in config_list:
        config = MyConfig(file_paths=config_files)

        runner = ModelRunner(config)
        runner.run(skip_if_exists=skip_if_exist)
        
        parser = ResponseParser(config=config)
        parser.run(skip_if_exist=skip_if_exist)
        break


if __name__ == "__main__":
    setup_log()
    main()
