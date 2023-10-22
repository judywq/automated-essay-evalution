from enum import Enum
import os
import logging
from lib.io import read_data

logger = logging.getLogger(__name__)


class Level(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Essay:
    TRUNCATE_LEN = 50

    def __init__(self, fn, text, prompt="", level=Level.NONE) -> None:
        self.fn = fn
        self.text = text
        self.prompt = prompt
        self.level = level

    def __str__(self) -> str:
        txt = (
            (self.text[: self.TRUNCATE_LEN] + "..")
            if len(self.text) > self.TRUNCATE_LEN
            else self.text
        )
        return f"Essay({self.fn}, {self.prompt}, {self.level.value}, {txt})"

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

    # @classmethod
    # def load_essays(cls, folder_path, ext='.txt'):
    #     essays = []
    #     for filename in os.listdir(folder_path):
    #         if filename.endswith(ext):
    #             with open(os.path.join(folder_path, filename)) as file:
    #                 e = cls(filename, file.read())
    #                 essays.append(e)
    #     return essays

    @classmethod
    def load_essays(cls, index_file, essay_root):
        def create_object(row):
            with open(os.path.join(essay_root, row["Filename"])) as file:
                text = file.read().strip()
            obj = cls(
                row["Filename"],
                text=text,
                prompt=row["Prompt"],
                level=row["Score Level"],
            )
            return obj

        df = read_data(index_file)
        objs = df.apply(create_object, axis=1).tolist()
        return objs
