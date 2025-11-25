# tts_hi.py
import torch
import nltk
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import soundfile as sf

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')  # POS tagging for English/Roman Hindi

# Preprocessing dictionary for common Roman Hindi words
ROMAN_HINDI_DICT = {
    "Main": "main",
    "main": "main",
    "hoon": "hoon",
    "mujhe": "mujhe",
    "pasand": "pasand",
    "hai": "hai",
    "Kal": "kal",
    "kal": "kal",
    "jaa": "jaa",
    "raha": "raha",
    "rahi": "rahi",
    "ho": "ho",
    "gaya": "gaya",
    "gayee": "gayee",
    "kar": "kar",
    "rahe": "rahe",
    "ki": "ki",
    "ke": "ke",
    "ko": "ko",
    "se": "se",
    "aur": "aur",
}

class HindiTTS:
    def __init__(self):
        print("Loading Hindi TTS model...")
        # Use language='indic' and speaker='v3_indic' for Hindi TTS
        self.model, _ = torch.hub.load(
            'snakers4/silero-models',
            'silero_tts',
            language='indic',
            speaker='v3_indic',
            trust_repo=True
        )
        print("Hindi TTS loaded!")

    def speak(self, text, speaker='hindi_female', outfile="hindi_output.wav"):
        """
        Convert Roman Hindi text to speech using Silero TTS.
        Proper nouns remain unchanged.
        """
        # Tokenize and POS-tag
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        processed_tokens = []
        for token, tag in pos_tags:
            # Keep proper nouns as-is
            if tag == 'NNP':
                processed_tokens.append(token)
            else:
                # Replace common Roman Hindi words using dictionary
                token_hi = ROMAN_HINDI_DICT.get(token, token)
                processed_tokens.append(token_hi)

        # Final processed Roman Hindi text
        text_hi = " ".join(processed_tokens)
        print(f"Processed Roman Hindi text for TTS: {text_hi}")

        # Generate audio
        audio = self.model.apply_tts(text=text_hi, speaker=speaker, sample_rate=48000)

        # Save audio
        sf.write(outfile, audio.cpu().numpy(), 48000)
        print(f"Saved audio to {outfile}")
