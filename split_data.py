import setting
from lib.io import read_data, write_data


def main():
    split(100, 10, 10)



def split(train_count, test_count, val_count):
    df = read_data(setting.index_path)
    print(df.head())
    df_sample = df.sample(n=train_count + test_count + val_count)
    df_train = df_sample.sample(n=train_count)
    df_sample.drop(df_train.index, inplace=True)
    df_test = df_sample.sample(n=test_count)
    df_val = df_sample.drop(df_test.index)
    write_data(df_train, "data/output/train.csv")
    write_data(df_test, "data/output/test.csv")
    write_data(df_val, "data/output/val.csv")


if __name__ == "__main__":
    main()
