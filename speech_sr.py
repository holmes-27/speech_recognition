import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Adjust this if you want more aggressive ambient noise reduction
with sr.Microphone(sample_rate=16000) as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Listening in real-time (Press Ctrl+C to stop)...\n")
    recognizer.pause_threshold = 1.0
    recognizer.phrase_threshold = 0.3
    recognizer.sample_rate = 48000
    recognizer.dynamic_energy_threshold = True
    recognizer.operation_timeout = 5
    recognizer.non_speaking_duration = 0.5
    recognizer.dynamic_energy_adjustment = 2
    recognizer.energy_threshold = 4000
    recognizer.phrase_time_limit = 10

    try:
        while True:
            print("Say something...")
            audio = recognizer.listen(source)

            try:
                # Use Google's Web Speech API
                text = recognizer.recognize_google(audio)
                print("You said:", text)
                
                if text.lower() in ["stop","bye","quit"]:
                    break

            except sr.UnknownValueError:
                print("Google could not understand audio.")
            except sr.RequestError as e:
                print("Could not request results; check your internet:", e)

    except KeyboardInterrupt:
        print("\nStopped by user.")
