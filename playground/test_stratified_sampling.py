import math
import pandas as pd

# Example DataFrame
data = {
    'Category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
    'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
df = pd.DataFrame(data)

def stratified_sampling():
    # Number of samples per group
    n_samples = 2

    # Perform stratified sampling
    sampled_df = df.groupby('Category', group_keys=False).apply(lambda x: x.sample(n=min(len(x), n_samples)))

    print(sampled_df)


def test_frac_sampling():
    
    df = pd.DataFrame({
        'A': [1,2],
    })
    frac = 0.9
    def calc_n_samples(frac, n):
        val = math.ceil(n * frac)
        if val == 0:
            val = n
        elif n - val < 1:
            val = n - 1
        print(f"n: {n}, frac: {frac}, val: {val}")
        return val

    print(df.sample(n=calc_n_samples(frac, len(df))))
    

if __name__ == "__main__":
    test_frac_sampling()