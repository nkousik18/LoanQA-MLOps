# grammar_english.py
from gramformer import Gramformer
gf = Gramformer(models=1, use_gpu=False)

def fix_english(text):
    result = gf.correct(text, max_candidates=1)
    return list(result)[0] if result else text
