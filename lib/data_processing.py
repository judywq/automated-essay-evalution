import pandas as pd
import setting
from lib.io import read_data, write_data

import logging
logger = logging.getLogger(__name__)


class DataSplitter:
    
    def __init__(self) -> None:
        self.max_num_per_group = {}
    
    def run(self, train_on_form=None):
        df = self.read_all_data()
        self.split_data(df, train_on_form=train_on_form)
        print("Data splitting done!")

    def read_all_data(self):
        form1_fn = setting.index_file_path_template.format(form_id=1)
        form2_fn = setting.index_file_path_template.format(form_id=2)

        df1 = read_data(form1_fn)
        df2 = read_data(form2_fn)
        
        self.calc_max_num_per_group(df1, df2)

        df1[setting.column_form] = 1
        df2[setting.column_form] = 2
        df = pd.concat([df1, df2], ignore_index=True)
        
        df[setting.column_item] = setting.item_id
                
        logger.info("Score shape:\n{}".format(df.value_counts([setting.column_form, setting.column_score]).sort_index()))

        return df
    
    def calc_max_num_per_group(self, df1, df2):
        value_counts_dict_df1 = df1[setting.column_score].value_counts().to_dict()
        value_counts_dict_df2 = df2[setting.column_score].value_counts().to_dict()

        # Combine dictionaries and calculate the minimum value for each key
        self.max_num_per_group = {key: min(value_counts_dict_df1.get(key, 0), value_counts_dict_df2.get(key, 0))
                        for key in set(value_counts_dict_df1) | set(value_counts_dict_df2)}
        
        logger.debug(f"Max num per group: {self.max_num_per_group}")

    def split_data(self, df, train_on_form=None):
        """Split data into train/val/test sets

        Args:
            df (pd.DataFrame): input data
            train_on_form (int, optional): train only on form N. Default (None) means both .
        """
        filter_integer = setting.integer_score_only
        df_copy = df.copy()
        df_train = self.sample(df_copy, num_per_group=setting.num_per_group["train"], filter_form=train_on_form, filter_integer=filter_integer)
        df_copy.drop(df_train.index, inplace=True)
        df_val = self.sample(df_copy, num_per_group=setting.num_per_group["val"], filter_form=train_on_form, filter_integer=filter_integer)
        df_copy.drop(df_val.index, inplace=True)
        df_test = self.sample(df_copy, num_per_group=setting.num_per_group["test"])

        write_data(df_train, setting.index_train_filename)
        write_data(df_test, setting.index_test_filename)
        write_data(df_val, setting.index_val_filename)
        
        print("Train shape:", df_train.shape)
        print("Val shape:", df_val.shape)
        print("Test shape:", df_test.shape)
    
    def sample(self, df: pd.DataFrame, num_per_group=-1, filter_form=None, filter_integer=False) -> pd.DataFrame:
        
        if filter_integer:
            df = df[df[setting.column_score] % 1 == 0].copy()

        if filter_form:
            df = df[df[setting.column_form] == filter_form]
        
        if num_per_group < 0:
            return df

        sampled_data = pd.concat([group.sample(min(num_per_group, self.max_num_per_group[key])) 
                                  for key, group in df.groupby(setting.column_score)])
        return sampled_data
