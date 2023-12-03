from enum import Enum
import os
import logging

import setting
from lib.io import read_data

logger = logging.getLogger(__name__)


class Level(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Essay:
    TRUNCATE_LEN = 50
    prompt_mapping = {}

    def __init__(
        self,
        fn,
        text,
        score,
        form_id=None,
        item_id=None,
        prompt="",
        prompt_text="",
        level=Level.NONE,
    ) -> None:
        self.fn = fn
        self.text = text
        self.score = score
        self.prompt = prompt
        self.prompt_text = prompt_text
        self.level = level
        self.form_id = form_id
        self.item_id = item_id

    def __str__(self) -> str:
        txt = (
            (self.text[: self.TRUNCATE_LEN] + "..")
            if len(self.text) > self.TRUNCATE_LEN
            else self.text
        )
        prompt_txt = (
            (self.prompt_text[: self.TRUNCATE_LEN] + "..")
            if len(self.prompt_text) > self.TRUNCATE_LEN
            else self.prompt_text
        )
        return f"Essay({self.fn},  S[{self.score}],  Q[{prompt_txt}], A[{txt})]"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        return {
            # "Sample_ID": self.fn,
            "ETS Score": self.score,
            "filename": self.fn,
            "essay_form_id": self.form_id,
            "essay_item_id": self.item_id,
            "essay_prompt": self.prompt_text,
            "essay": self.text,
        }

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
    def load_essays_from_file(cls, index_file):
        df = read_data(index_file)
        return cls.load_essays_from_dataframe(df)

    @classmethod
    def load_essays_from_dataframe(cls, df):
        def create_object(row):
            response_fn = get_response_fn(row)
            with open(response_fn) as file:
                text = file.read().strip()

            prompt_text = get_prompt_text(row)

            obj = cls(
                response_fn,
                text=text,
                score=get_score(row),
                prompt_text=prompt_text,
                form_id=row[setting.column_form],
                item_id=row[setting.column_item],
            )
            return obj

        objs = df.apply(create_object, axis=1).tolist()
        return objs


def get_response_fn(row):
    response_fn = setting.response_file_path_tempalte.format(
        form_id=row[setting.column_form],
        response_id=row[setting.column_response_id],
        item_id=row[setting.column_item],
    )
    return response_fn


prompt_mapping = {}


def get_prompt_text(row):
    prompt_fn = setting.prompt_file_path_tempalte.format(
        form_id=row[setting.column_form],
        item_id=row[setting.column_item],
    )
    prompt_text = prompt_mapping.get(prompt_fn, None)
    if not prompt_text:
        with open(prompt_fn) as file:
            prompt_text = file.read().strip()
            prompt_mapping[prompt_fn] = prompt_text
    return prompt_text


def get_score(row):
    return row[setting.column_score]
