from lib.utils import setup_log
from lib.model_runner import ModelRunner
from lib.data_processing import ResponseParser
from lib.config import MyConfig
from configlist import config_list


def main():
    for config_files in config_list:
        config = MyConfig(file_paths=config_files)

        runner = ModelRunner(config)
        runner.run()
        
        parser = ResponseParser(config=config)
        parser.run()
        # break


if __name__ == "__main__":
    setup_log()
    main()
