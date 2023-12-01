import json
from time import sleep
import pandas as pd
from lib.chat import MyBotWrapper
from lib.parser import EssayEvaluationWithTunedModelParser
from lib.essay import Essay
from lib.io import write_data
from lib.utils import setup_log, calc_and_write_success_rate
import setting

import logging

logger = logging.getLogger(__name__)


def main():
    # system_message = setting.system_message
    system_message = setting.system_message_short
    num_limits = 999

    resp = json.load(open(setting.job_id_filename, "r"))
    fine_tuned_model_id = resp["fine_tuned_model"]
    print(fine_tuned_model_id)

    essays = Essay.load_essays(index_file=setting.index_test_filename)

    bot = MyBotWrapper(
        parser=EssayEvaluationWithTunedModelParser(),
        model=fine_tuned_model_id,
        temperature=0,
    )

    results = []
    for index, essay in enumerate(essays):
        logger.info(f"Processing essay {index + 1}/{len(essays)}...")

        res = bot.run(inputs={"system_message": system_message, "essay": essay})
        result = res["result"]
        results.append(result)

        if index + 1 >= num_limits:
            break

        if index % 10 == 0:
            df = pd.DataFrame(results)
            write_data(df, setting.test_result_filename)

        if index % 60 == 0:
            sleep(10)

    df = pd.DataFrame(results)
    write_data(df, setting.test_result_filename)
    
    calc_and_write_success_rate(setting.test_result_filename, threshold=0.5)
    
    print("Done!")


if __name__ == "__main__":
    setup_log()
    main()
