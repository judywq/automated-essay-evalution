from lib.chat import MyBotWrapper
from lib.model_runner_old import run_model
from lib.parser import EssayEvaluationScoreParser
from lib.utils import setup_log
import setting

import logging

logger = logging.getLogger(__name__)


def main():
    index_file = setting.index_test_filename
    system_message = setting.system_message
    test_result_filename = setting.test_result_official_filename

    bot = MyBotWrapper(
        parser=EssayEvaluationScoreParser(),
        model=setting.DEFAULT_MODEL,
        temperature=0,
    )

    run_model(
        index_file=index_file,
        bot=bot,
        system_message=system_message,
        test_result_filename=test_result_filename,
    )


if __name__ == "__main__":
    setup_log()
    main()
