# # test_voice.py
# from tts_hi import HindiTTS
#
# def main():
#     hi_tts = HindiTTS()
#
#     speakers = [
#         'bengali_female', 'bengali_male', 'gujarati_female', 'gujarati_male',
#         'hindi_female', 'hindi_male', 'kannada_female', 'kannada_male',
#         'malayalam_female', 'malayalam_male', 'manipuri_female', 'rajasthani_female',
#         'rajasthani_male', 'tamil_female', 'tamil_male', 'telugu_female',
#         'telugu_male', 'random'
#     ]
#
#     # Show available speakers
#     print("Available Hindi speakers:")
#     for idx, sp in enumerate(speakers, 1):
#         print(f"{idx}. {sp}")
#
#     # Select speaker
#     while True:
#         try:
#             sp_num = int(input("Select a speaker (number): "))
#             if 1 <= sp_num <= len(speakers):
#                 break
#             else:
#                 print(f"Enter a number between 1 and {len(speakers)}")
#         except ValueError:
#             print("Enter a valid number.")
#
#     speaker = speakers[sp_num - 1]
#
#     # Input text (Roman Hindi only)
#     text = input("Enter text to speak in Hindi (Roman Hindi only, proper nouns allowed): ")
#
#     # Generate TTS
#     hi_tts.speak(text, speaker=speaker, outfile="hindi_test.wav")
#
# if __name__ == "__main__":
#     main()

# test_english_tts.py
import torch
import soundfile as sf

# Load Silero English TTS
model, _ = torch.hub.load(
    'snakers4/silero-models',
    'silero_tts',
    language='en',   # English
    speaker='v3_en_indic',
    trust_repo=True
)

# Input English text
text = "The borrower, Pratyusha PVR, has applied for a personal loan of ₹5,00,000. The loan tenure is 24 months, and the interest rate is 10% per annum."

# Generate audio
audio = model.apply_tts(text=text, speaker='v3_en_indic', sample_rate=48000)

# Save audio
sf.write("english_test.wav", audio.cpu().numpy(), 48000)
print("Saved english_test.wav")
