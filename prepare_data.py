from pprint import pprint
from icecream import ic
from lib.essay import Essay

sample_10_root = './data/input/responses/sample-10'

def prepare():
    essays = Essay.load_essays(sample_10_root)
    ic(essays)
    # print(essays)


if __name__ == '__main__':
    prepare()
