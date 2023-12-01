import pandas as pd
import setting
from lib.io import read_data, write_data


def main():
    df = read_all_data()
    # train_on_form = 1
    # train_on_form = 2
    train_on_form = None
    split_data(df, train_on_form=train_on_form)
    print("Done!")


def read_all_data():
    form1_fn = setting.index_file_path_template.format(form_id=1)
    form2_fn = setting.index_file_path_template.format(form_id=2)

    df1 = read_data(form1_fn)
    df2 = read_data(form2_fn)

    df1[setting.column_form] = 1
    df2[setting.column_form] = 2
    df = pd.concat([df1, df2], ignore_index=True)
    
    df[setting.column_item] = setting.item_id
    return df


def split_data(df, train_on_form=None):
    """Split data into train/val/test sets

    Args:
        df (pd.DataFrame): input data
        train_on_form (int, optional): train only on form N. Defaults to None.
    """
    df_copy = df.copy()
    df_train = sample(df_copy, num_per_group=setting.num_per_group["train"], filter_form=train_on_form)
    df_copy.drop(df_train.index, inplace=True)
    df_test = sample(df_copy, num_per_group=setting.num_per_group["test"])
    df_copy.drop(df_test.index, inplace=True)
    df_val = sample(df_copy, num_per_group=setting.num_per_group["val"], filter_form=train_on_form)

    write_data(df_train, setting.index_train_filename)
    write_data(df_test, setting.index_test_filename)
    write_data(df_val, setting.index_val_filename)
    
    print("Train shape:", df_train.shape)
    print("Val shape:", df_val.shape)
    print("Test shape:", df_test.shape)


def print_stats():
    df = read_all_data()
    print(df[setting.column_score].value_counts())


def sample(df: pd.DataFrame, num_per_group=10, filter_form=None) -> pd.DataFrame:
    # sample n records from each group (if n > group size, take all records in the group)
    def sample_group(group, num_elements):
        return group.sample(min(num_elements, len(group)), replace=False)
    
    if filter_form:
        df = df[df[setting.column_form] == filter_form]

    sampled_data = df.groupby(
        [setting.column_score], group_keys=False, as_index=False
    ).apply(sample_group, num_elements=num_per_group)

    return sampled_data


if __name__ == "__main__":
    main()
    # print_stats()
