# src/data_acquisition/vosk_function.py

import pyaudio
from vosk import Model, KaldiRecognizer
import json

class VoskRecognizer:
    def __init__(self, model_path):
        try:
            self.model = Model(model_path)
            print(f"✅ Modèle Vosk '{model_path}' initialisé.")
        except Exception as e:
            print(f"ERREUR Chargement Vosk: {e}")
            raise

    def start_transcription(self, callback_function, audio_source_iterator=None):
        """
        Démarre la transcription.
        :param callback_function: Fonction appelée quand du texte est détecté.
        :param audio_source_iterator: (Optionnel) Un itérateur qui yield des bytes d'audio (ex: pour ROS).
                                      Si None, utilise le micro local via PyAudio.
        """
        recognizer = KaldiRecognizer(self.model, 16000)
        
        if audio_source_iterator:
            # --- MODE FLUX EXTERNE (ROS) ---
            print(">>> Transcription sur flux externe (ROS) démarrée...")
            try:
                for data in audio_source_iterator():
                    if len(data) == 0: continue
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "")
                        if text: callback_function(text)
            except Exception as e:
                print(f"Erreur transcription externe: {e}")

        else:
            # --- MODE MICRO LOCAL (PyAudio) ---
            p = pyaudio.PyAudio()
            try:
                stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                                input=True, frames_per_buffer=8192)
                print(">>> Transcription sur Micro Local démarrée...")
                
                while True:
                    data = stream.read(4096, exception_on_overflow=False)
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "")
                        if text: callback_function(text)
            except Exception as e:
                print(f"Erreur transcription locale: {e}")
            finally:
                if 'stream' in locals() and stream.is_active():
                    stream.stop_stream()
                    stream.close()
                p.terminate()