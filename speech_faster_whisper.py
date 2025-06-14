import os
import wave
import pyaudio
from faster_whisper import WhisperModel

# Initialize Whisper
model = WhisperModel("small.en", device="cpu", compute_type="int8")

# Audio configuration
RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 5
TEMP_FILE = "temp_audio.wav"

# Setup audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def record_audio():
    print("Recording...")
    frames = []
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    wf = wave.open(TEMP_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio():
    segments, _ = model.transcribe(TEMP_FILE)
    transcription = " ".join(segment.text for segment in segments)
    return transcription

try:
    while True:
        record_audio()
        text = transcribe_audio()
        print("Transcribed:", text)
        os.remove(TEMP_FILE)

except KeyboardInterrupt:
    print("\nStopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
