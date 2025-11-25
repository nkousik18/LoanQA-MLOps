# interactive_tts.py
from tts_en import EnglishTTS
from tts_hi import HindiTTS

def main():
    print("Welcome to TTS Assistant!")
    en_tts = EnglishTTS()
    hi_tts = HindiTTS()

    while True:
        lang = input("Choose language (en/hi) or 'exit' to quit: ").strip().lower()
        if lang == "exit":
            break
        text = input("Enter text: ").strip()
        outfile = input("Enter output filename (default based on language): ").strip()
        if not outfile:
            outfile = f"{lang}_output.wav"

        if lang == "en":
            en_tts.speak(text, outfile)
        elif lang == "hi":
            hi_tts.speak(text, outfile)
        else:
            print("Invalid language. Choose 'en' or 'hi'.")

if __name__ == "__main__":
    main()
