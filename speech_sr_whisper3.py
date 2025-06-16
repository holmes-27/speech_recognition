# Only records when it hears voice

import io
import speech_recognition as sr
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# Initialize Whisper model
model = WhisperModel("small.en", device="cpu", compute_type="int8")

# Setup recognizer
recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.8
recognizer.non_speaking_duration = 0.5
recognizer.dynamic_energy_threshold = True
recognizer.energy_threshold = 300  # Can tweak this

mic = sr.Microphone(sample_rate=16000)

print("Calibrating for ambient noise...")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Speak to start. (Ctrl+C to stop)\n")

    try:
        while True:
            print("Waiting for speech...")
            audio = recognizer.listen(source)

            wav_data = audio.get_wav_data()
            audio_bytes = io.BytesIO(wav_data)
            audio_array, samplerate = sf.read(audio_bytes)

            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            segments, _ = model.transcribe(audio_array, language="en")
            text = " ".join(segment.text for segment in segments)
            if text.strip():
                print("Transcription:", text.strip())
            else:
                print("Could not transcribe speech.")

    except KeyboardInterrupt:
        print("\nStopped.")
