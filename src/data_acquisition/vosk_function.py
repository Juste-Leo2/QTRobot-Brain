# src/data_acquisition/vosk_function.py

import pyaudio
from vosk import Model, KaldiRecognizer
import json

class VoskRecognizer:
    """
    Une classe pour gérer la reconnaissance vocale avec Vosk.
    """
    def __init__(self, model_path):
        """
        Initialise le reconnaisseur vocal.
        :param model_path: Chemin vers le dossier du modèle Vosk.
        """
        try:
            self.model = Model(model_path)
            print(f"✅ Modèle Vosk '{model_path}' initialisé avec succès.")
        except Exception as e:
            print(f"ERREUR: Impossible de charger le modèle depuis '{model_path}'.")
            print(e)
            raise

    def start_transcription(self, callback_function, device_index=None):
        """
        Démarre la transcription en temps réel et appelle le callback avec le texte reconnu.
        Cette fonction est bloquante et est destinée à être exécutée dans un thread.
        """
        recognizer = KaldiRecognizer(self.model, 16000)
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=16000,
                            input=True,
                            frames_per_buffer=8192,
                            input_device_index=device_index)
            
            print(">>> Prêt à écouter...")
            
            while True: # Note: This is an infinite loop, hard to test directly
                data = stream.read(4096, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        callback_function(text)

        except Exception as e:
            print(f"Une erreur est survenue lors de la transcription: {e}")
        finally:
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()
            p.terminate()
            print("🛑 Transcription arrêtée et ressources nettoyées.")

# Code d'exemple pour exécuter ce module seul
if __name__ == '__main__':
    def my_callback(text):
        print(f"Texte reconnu: {text}")

    # Mettez ici le chemin vers votre modèle pour un test rapide
    # Idéalement, ce chemin viendrait d'un fichier de configuration
    MODEL_PATH = "../../models/stt_vosk/vosk-model-small-fr-0.22"
    
    try:
        recognizer = VoskRecognizer(MODEL_PATH)
        recognizer.start_transcription(my_callback)
    except Exception as e:
        print(f"Impossible de lancer la démo. Assurez-vous que le chemin du modèle est correct. Erreur: {e}")