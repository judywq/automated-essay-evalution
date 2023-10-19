import spacy
from icecream import ic
from spacy_readability import Readability

def test_readability():
    nlp = spacy.load('en')
    # nlp = spacy.load("en_core_web_sm")
    read = Readability()
    nlp.add_pipe(read, last=True)
    doc = nlp("I am some really difficult text to read because I use obnoxiously large words.")
    ic(doc._.flesch_kincaid_grade_level)
    ic(doc._.flesch_kincaid_reading_ease)
    ic(doc._.dale_chall)


if __name__ == '__main__':
    test_readability()
