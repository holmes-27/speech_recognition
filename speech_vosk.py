# Download the model: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

import sounddevice as sd
import queue
import vosk
import json

q = queue.Queue()
model = vosk.Model("vosk-model-small-en-us-0.15")

def callback(indata, frames, time, status):
    q.put(bytes(indata))

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    rec = vosk.KaldiRecognizer(model, 16000)
    print("Say something (say 'stop' to exit):")

    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            print("> ", text)
            if "stop" in text.lower():
                print("Stopping...")
                break