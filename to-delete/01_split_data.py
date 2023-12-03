from lib.utils import setup_log
from lib.data_processing import DataSplitter
import setting


def main():
    splitter = DataSplitter()
    splitter.run(train_on_form=setting.train_on_form)


if __name__ == "__main__":
    setup_log()
    main()
    # print_stats()
