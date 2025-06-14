import os
import wave
import pyaudio
from faster_whisper import WhisperModel

# Setup model
model_size = "small.en" 
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Audio settings
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
TEMP_WAV = "temp_chunk.wav"

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

accumulated_transcription = ""

def record_chunk(duration=4):  # duration in seconds
    print("Recording chunk...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    wf = wave.open(TEMP_WAV, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_chunk():
    segments, _ = model.transcribe(TEMP_WAV)
    return " ".join(segment.text for segment in segments)

try:
    while True:
        record_chunk()
        text = transcribe_chunk()
        print(text)
        accumulated_transcription += text + " "
        os.remove(TEMP_WAV)

except KeyboardInterrupt:
    print("\nStopping... Saving transcript.")
    with open("log.txt", "w") as f:
        f.write(accumulated_transcription)

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
