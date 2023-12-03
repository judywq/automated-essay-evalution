from lib.data_processing import DatasetPreparation
from lib.utils import setup_log
import setting


def main():
    prepare = DatasetPreparation()
    prepare.run()


if __name__ == "__main__":
    setup_log()
    main()
