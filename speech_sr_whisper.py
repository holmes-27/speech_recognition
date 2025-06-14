# Record voice with speech_recognition, save the temp file and use the file to get the text using Faster Whisper

import speech_recognition as sr
from faster_whisper import WhisperModel
import os

# Setup Whisper model
model = WhisperModel("small.en", device="cpu", compute_type="int8")

# Setup recognizer and microphone
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16000)

TEMP_FILE = "temp_sr_audio.wav"

print("🎙️ Calibrating for ambient noise...")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Ready! Start speaking... (press Ctrl+C to stop)\n")

    try:
        while True:
            print("Listening for speech...")
            audio = recognizer.listen(source, phrase_time_limit=5)

            # Save to WAV file
            with open(TEMP_FILE, "wb") as f:
                f.write(audio.get_wav_data())

            print("Transcribing with Whisper...")
            segments, _ = model.transcribe(TEMP_FILE)
            text = " ".join(segment.text for segment in segments)
            print("Transcription:", text.strip())

            os.remove(TEMP_FILE)

    except KeyboardInterrupt:
        print("\nExiting...")
