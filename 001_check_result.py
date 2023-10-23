from lib.io import read_data

fn = "data/output/2023-10-23-update-rubrics/results/test-result-2023-10-23-19-07-11.xlsx"

df = read_data(fn)
success_rate = df["ok"].sum() / len(df)
print("Done! Success rate:", success_rate)
