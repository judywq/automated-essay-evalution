import os

input_root= 'data/input/AWE'
index_path = os.path.join(input_root, 'index', 'index.csv')

output_root = 'data/output'
train_path = os.path.join(output_root, 'train.csv')
val_path = os.path.join(output_root, 'val.csv')
test_path = os.path.join(output_root, 'test.csv')

num_per_group = {
    'train': 30,
    'val': 10,
    'test': 10
}