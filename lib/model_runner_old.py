from time import sleep
import pandas as pd
from lib.essay import Essay
from lib.io import write_data
from lib.utils import calc_and_write_success_rate


import logging
logger = logging.getLogger(__name__)


def run_model(index_file, bot, system_message, test_result_filename, num_limits=999):
    essays = Essay.load_essays_from_file(index_file=index_file)

    results = []
    for index, essay in enumerate(essays):
        logger.info(f"Processing essay {index + 1}/{len(essays)}...")

        res = bot.run(inputs={"system_message": system_message, "essay": essay})
        success = res["success"]
        if not success:
            logger.error(f"+++ Failed to process essay {index + 1}/{len(essays)}. Raw response:\n {res['raw_response']}")
            continue
        result = res["result"]
        results.append(result)

        if index + 1 >= num_limits:
            break

        if index % 10 == 0:
            df = pd.DataFrame(results)
            write_data(df, test_result_filename)

        if index % 60 == 0:
            sleep(10)

    df = pd.DataFrame(results)
    write_data(df, test_result_filename)
    calc_and_write_success_rate(test_result_filename)
    
    print("Done!")
