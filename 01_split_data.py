import pandas as pd
import setting
from lib.io import read_data, write_data


def main():
    df = read_data(setting.index_path)
    split_data(df)
    print("Done!")


def split_data(df):
    df_copy = df.copy()
    df_train = sample(df_copy, num_per_group=setting.num_per_group["train"])
    df_copy.drop(df_train.index, inplace=True)
    df_test = sample(df_copy, num_per_group=setting.num_per_group["test"])
    df_copy.drop(df_test.index, inplace=True)
    df_val = sample(df_copy, num_per_group=setting.num_per_group["val"])

    write_data(df_train, setting.index_train_filename)
    write_data(df_test, setting.index_test_filename)
    write_data(df_val, setting.index_val_filename)
    
    print("Train shape:", df_train.shape)
    print("Val shape:", df_val.shape)
    print("Test shape:", df_test.shape)


def print_stats():
    df = read_data(setting.index_path)
    print(df[(df["Score Level"] == "low")]["Language"].value_counts())
    print(df[(df["Score Level"] == "medium")]["Language"].value_counts())
    print(df[(df["Score Level"] == "high")]["Language"].value_counts())

    print(df.groupby(["Prompt", "Score Level"], group_keys=False).count())


def sample(df: pd.DataFrame, num_per_group=30):
    # Define a function to sample records from each language within a group
    def sample_group(group):
        df_base = group.copy()
        sampled_records = []
        count = 0
        while count < num_per_group:
            tmp = df_base.groupby("Language", group_keys=False).apply(
                lambda x: x.sample(n=1)
            )
            if tmp.shape[0] + count > num_per_group:
                tmp = tmp.sample(num_per_group - count)
            count += tmp.shape[0]
            sampled_records.append(tmp)
            df_base = df_base.drop(pd.concat(sampled_records).index, errors="ignore")
        return pd.concat(sampled_records)

    # Group by 'Prompt', 'Score Level', then sample records from each language within each group
    sampled_data = df.groupby(
        ["Prompt", "Score Level"], group_keys=False, as_index=False
    ).apply(sample_group)

    # print(sampled_data.groupby(['Prompt', 'Score Level'], group_keys=False).count())
    # print(sampled_data.groupby(['Prompt', 'Score Level', 'Language']).count())
    # Print the sampled data
    # print(sampled_data[(sampled_data['Score Level'] == 'low') & (sampled_data['Prompt'] == 'P2')]['Language'].value_counts())
    return sampled_data


if __name__ == "__main__":
    main()
