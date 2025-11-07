import difflib

def char_accuracy(gt: str, pred: str) -> float:
    sm = difflib.SequenceMatcher(None, gt, pred)
    return sm.ratio()*100

if __name__=="__main__":
    with open("reference.txt","r",encoding="utf-8") as f: gt=f.read()
    with open("pred.txt","r",encoding="utf-8") as f: pr=f.read()
    print(f"OCR_accuracy: {char_accuracy(gt, pr):.1f}%")
