from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)


class Level(Enum):
    NONE = 'none'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


class Essay:
    TRUNCATE_LEN = 30
    def __init__(self, fn, text) -> None:
        self.fn = fn
        self.text = text
        self.level = Level.NONE
    
    def __str__(self) -> str:
        txt = (self.text[:self.TRUNCATE_LEN] + '..') if len(self.text) > self.TRUNCATE_LEN else self.text
        return f"Essay({self.fn}, {txt})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def level(self):
        return self._level
    
    @level.setter
    def level(self, level):
        if isinstance(level, str):
            try:
                level = Level(level)
            except ValueError:
                logger.warning(f"Invalid essay level: {level}")
                level = Level.NONE
        self._level = level

    @classmethod
    def load_essays(cls, folder_path, ext='.txt'):
        essays = []
        for filename in os.listdir(folder_path):
            if filename.endswith(ext):
                with open(os.path.join(folder_path, filename)) as file:
                    e = cls(filename, file.read())
                    essays.append(e)
        return essays
