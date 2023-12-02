import json
from lib.chat import MyBotWrapper
from lib.model_runner import run_model
from lib.parser import EssayEvaluationWithTunedModelParser
from lib.utils import setup_log
import setting

import logging

logger = logging.getLogger(__name__)


def main():
    index_file = setting.index_test_filename
    # system_message = setting.system_message
    system_message = setting.system_message_short
    test_result_filename = setting.test_result_tuned_filename

    resp = json.load(open(setting.job_id_filename, "r"))
    fine_tuned_model_id = resp["fine_tuned_model"]
    print(fine_tuned_model_id)

    bot = MyBotWrapper(
        parser=EssayEvaluationWithTunedModelParser(),
        model=fine_tuned_model_id,
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
