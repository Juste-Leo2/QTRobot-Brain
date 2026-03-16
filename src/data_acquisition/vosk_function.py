# src/data_acquisition/vosk_function.py

import pyaudio
from vosk import Model, KaldiRecognizer
import json

class VoskRecognizer:
    def __init__(self, models_dict):
        """
        :param models_dict: dict of language code to model path, e.g., {'fr': 'path_fr', 'en': 'path_en'}
        """
        self.models = {}
        try:
            for lang, path in models_dict.items():
                self.models[lang] = Model(path)
                print(f"✅ Modèle Vosk ({lang}) '{path}' initialisé.")
        except Exception as e:
            print(f"ERREUR Chargement Vosk: {e}")
            raise

    def start_transcription(self, callback_function, audio_source_iterator=None, pause_checker=None, language_getter=None):
        """
        Démarre la transcription.
        :param callback_function: Fonction appelée quand du texte est détecté.
        :param audio_source_iterator: (Optionnel) Un itérateur qui yield des bytes d'audio (ex: pour ROS).
                                      Si None, utilise le micro local via PyAudio.
        :param pause_checker: (Optionnel) Fonction retournant True si on doit ignorer l'audio.
        :param language_getter: (Optionnel) Fonction retournant la langue actuelle ('fr', 'en'...).
        """
        current_lang = language_getter() if language_getter else list(self.models.keys())[0]
        recognizer = KaldiRecognizer(self.models[current_lang], 16000)
        print(f"🎙️ Ecoute réglée sur la langue: {current_lang}")
        
        if audio_source_iterator:
            # --- MODE FLUX EXTERNE (ROS) ---
            print(">>> Transcription sur flux externe (ROS) démarrée...")
            try:
                for data in audio_source_iterator():
                    if language_getter:
                        new_lang = language_getter()
                        if new_lang != current_lang and new_lang in self.models:
                            current_lang = new_lang
                            recognizer = KaldiRecognizer(self.models[current_lang], 16000)
                            print(f"🎙️ STT (ROS) a switché sur la langue: {current_lang}")
                            
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
                    
                    if language_getter:
                        new_lang = language_getter()
                        if new_lang != current_lang and new_lang in self.models:
                            current_lang = new_lang
                            recognizer = KaldiRecognizer(self.models[current_lang], 16000)
                            print(f"🎙️ STT (Local) a switché sur la langue: {current_lang}")

                    if pause_checker and pause_checker():
                        continue
                        
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