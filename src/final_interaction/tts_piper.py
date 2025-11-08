# src/final_interaction/tts_piper.py

import wave
from piper import PiperVoice, SynthesisConfig
import os

class PiperTTS:
    """
    Une classe pour gérer la synthèse vocale avec Piper.
    Charge un modèle une seule fois et permet de synthétiser du texte avec différentes voix.
    """
    def __init__(self, model_path: str):
        """
        Initialise le synthétiseur vocal en chargeant le modèle.
        :param model_path: Chemin vers le fichier modèle .onnx.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier modèle Piper n'a pas été trouvé : {model_path}")
        
        print(f"Chargement du modèle Piper depuis '{model_path}'...")
        self.voice = PiperVoice.load(model_path)
        print("✅ Modèle Piper chargé.")

    def synthesize(self, text: str, output_path: str, speaker_id: int = 0, length_scale=1.0, noise_scale=0.667):
        """
        Synthétise le texte donné et l'enregistre dans un fichier WAV.
        :param text: Le texte à synthétiser.
        :param output_path: Le chemin du fichier WAV de sortie.
        :param speaker_id: L'ID du locuteur à utiliser (pour les modèles multi-locuteurs).
        :param length_scale: Vitesse de la parole ( > 1 plus lent, < 1 plus rapide).
        :param noise_scale: Variabilité de la parole.
        """
        synthesis_config = SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale
        )
        with wave.open(output_path, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file, syn_config=synthesis_config)
        print(f"Audio synthétisé et enregistré dans '{output_path}'")


# Code d'exemple pour exécuter ce module seul
if __name__ == '__main__':
    # Chemin vers le modèle (à adapter ou charger depuis une config)
    MODEL_PATH = "../../models/tts_piper/fr_FR-upmc-medium.onnx"
    
    try:
        tts = PiperTTS(MODEL_PATH)

        # Test voix 1 (Jessica)
        tts.synthesize(
            text="Bonjour, je suis la voix de Jessica. Armand est un beau gosse.",
            output_path="test_jessica.wav",
            speaker_id=0
        )

        # Test voix 2 (Pierre)
        tts.synthesize(
            text="Et moi, je suis la voix de Pierre. Je confirme qu'Armand est charismatique.",
            output_path="test_pierre.wav",
            speaker_id=1,
            length_scale=1.1 # Un peu plus lent
        )

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")