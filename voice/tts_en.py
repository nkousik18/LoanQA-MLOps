# import torch
# import soundfile as sf
#
# class VoiceAssistant:
#     def __init__(self, speaker="en_0"):
#         # Load Silero TTS model
#         self.speaker = speaker
#         self.model, _ = torch.hub.load(
#             repo_or_dir="snakers4/silero-models",
#             model="silero_tts",
#             language="en",
#             speaker="v3_en"
#         )
#
#
#     def speak(self, text, outfile="tts_output.wav"):
#         audio = self.model.apply_tts(
#             text=text,
#             speaker=self.speaker,
#             sample_rate=48000
#         )
#         sf.write(outfile, audio, 48000)
#         return outfile
#
# import torch
# import soundfile as sf
# from gtts import gTTS
#
# class VoiceAssistant:
#     def __init__(self, lang="en", speaker=None):
#         self.lang = lang
#
#         if lang == "en":
#             # Silero English
#             if speaker is None:
#                 speaker = "en_0"  # valid English speaker
#             self.speaker = speaker
#             self.model, _ = torch.hub.load(
#                 repo_or_dir="snakers4/silero-models",
#                 model="silero_tts",
#                 language="en",
#                 speaker=speaker
#             )
#         elif lang == "hi":
#             # For Hindi, we'll use gTTS
#             self.speaker = "hi"
#
#     def speak(self, text, outfile="tts_output.wav"):
#         if self.lang == "en":
#             audio = self.model.apply_tts(
#                 text=text,
#                 speaker=self.speaker,
#                 sample_rate=48000
#             )
#             sf.write(outfile, audio, 48000)
#         elif self.lang == "hi":
#             tts = gTTS(text=text, lang="hi")
#             tts.save(outfile)
#         return outfile

import torch
from scipy.io.wavfile import write

class EnglishTTS:
    def __init__(self, speaker="v3_en_indic"):
        print("Loading English TTS model...")
        self.model, _ = torch.hub.load(
            "snakers4/silero-models",
            "silero_tts",
            "v3_en",  # positional model name
            speaker=speaker
        )
        print("English TTS loaded!")

    def speak(self, text, outfile="english.wav"):
        audio = self.model.apply_tts(text=text, speaker="v3_en_indic", sample_rate=48000)
        write(outfile, 48000, audio)
        print(f"Saved English audio to {outfile}")
