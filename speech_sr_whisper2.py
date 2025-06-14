# speech_recognition + whisper without the need for the temp file

import io
import speech_recognition as sr
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# Initialize Whisper model
model = WhisperModel("small.en", device="cpu", compute_type="int8")

# Setup recognizer and microphone
recognizer = sr.Recognizer()
#recognizer.pause_threshold = 1.0
#recognizer.phrase_threshold = 0.3
#recognizer.non_speaking_duration = 0.5
#recognizer.dynamic_energy_threshold = True

mic = sr.Microphone(sample_rate=16000)

print("Calibrating for ambient noise...")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Ready! Speak naturally... (Ctrl+C to stop)\n")

    try:
        while True:
            print("Listening...")
            audio = recognizer.listen(source, phrase_time_limit=5)

            # Convert AudioData to numpy array
            wav_data = audio.get_wav_data()
            audio_bytes = io.BytesIO(wav_data)

            # Load audio using soundfile
            audio_array, samplerate = sf.read(audio_bytes)

            # Whisper expects 16kHz mono float32 PCM
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)  # Convert stereo to mono

            # Transcribe using Whisper
            segments, _ = model.transcribe(audio_array, language="en")
            text = " ".join(segment.text for segment in segments)
            print("Transcription:", text.strip())

    except KeyboardInterrupt:
        print("\nStopped by user.")
