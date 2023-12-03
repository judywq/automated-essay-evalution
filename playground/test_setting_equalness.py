from lib.utils import setup_log
from lib.data_processing import DataSplitter, DatasetPreparation
from lib.config import MyConfig
import setting


def main():
    config_base = "./configs/config.base.json"
    config_form_1 = "./configs/config.form.1.json"
    config_form_2 = "./configs/config.form.2.json"
    config_form_all = "./configs/config.form.all.json"
    config_gpt_3_5_turbo = "./configs/config.gpt3.5-turbo.json"
    config_gpt_4_turbo = "./configs/config.gpt4-turbo.json"
    config_int = "./configs/config.type.int.json"
    config_float = "./configs/config.type.float.json"
    
    
    config_files = [config_base, config_form_1, config_float, config_gpt_4_turbo]
    config = MyConfig()
    config.load_configs(file_paths=config_files)
    
    for k in [
        'train_on_form',
        'tof_name',
        'integer_score_only',
        'global_run_id',
        'model_suffix',
        'test_result_prefix',
        'index_file_path_template',
        'output_root',
        'test_result_finetuned_filename',
        'num_per_group',
        'system_message_short',
    ]:
        if getattr(config, k) != getattr(setting, k):
            print(f'{k} is not equal')
            print(f'config: {getattr(config, k)}')
            print(f'setting: {getattr(setting, k)}')
            print()



if __name__ == "__main__":
    setup_log()
    main()
