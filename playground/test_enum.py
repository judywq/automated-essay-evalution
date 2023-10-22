
from enum import Enum


class Level(Enum):
    NONE = 'none'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


def test_enum():
    level = Level('low')
    print(level)
    
    try:
        level = Level('non-exist')
        print(level)
    except ValueError as e:
        print(e)



if __name__ == '__main__':
    test_enum()
    