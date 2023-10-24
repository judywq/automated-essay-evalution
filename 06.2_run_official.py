from time import sleep
from icecream import ic
import pandas as pd
import setting
from lib.io import write_data
from lib.chat import MyBotWrapper
from lib.essay import Essay
from lib.parser import EssayEvaluationParser, EssayEvaluationScoreParser
from lib.utils import setup_log

import logging
logger = logging.getLogger(__name__)

def main():
    num_limits = 1000
    
    essays = Essay.load_essays(
        index_file=setting.index_test_filename,
        essay_root=setting.essay_root,
        prompt_root=setting.essay_prompt_root,
    )
    bot = MyBotWrapper(
        # parser=EssayEvaluationParser(), 
        parser=EssayEvaluationScoreParser(),
        model=setting.DEFAULT_MODEL, 
        temperature=0
    )
    
    results = []
    for index, essay in enumerate(essays):
        logger.info(f"Processing essay {index + 1}/{len(essays)}...")
        ic(essay)
        
        res = bot.run(inputs={
            # "system_message": setting.system_message_level,
            "system_message": setting.system_message_score,
            "essay": essay,
        })
        
        # ic(res)
        results.append(res['result'])

        df_data = pd.DataFrame(results)
        write_data(df_data, setting.test_result_filename)

        if index + 1 >= num_limits:
            break
        
        if (index + 1) % 60 == 0:
            logger.info(f"Sleeping for 10 seconds...")
            sleep(10)
    
    df_data = pd.DataFrame(results)
    write_data(df_data, setting.test_result_filename)

    success_rate = df_data["ok"].sum() / len(df_data)
    with open(setting.test_result_filename + ".txt", "a") as f:
        f.write(f"Success rate: {success_rate}\n")
    
    print("Done! Success rate:", success_rate)

if __name__ == "__main__":
    setup_log()
    main()
