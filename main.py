# main.py
import threading
import yaml
import sys
import cv2      
import time
import argparse
import subprocess
from PIL import Image

# --- FIX ENCODAGE WINDOWS (CRITIQUE POUR CI/GITHUB ACTIONS) ---
# Force l'UTF-8 pour éviter les erreurs avec les émojis (🧪, 🤖, etc.)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# --- Imports de tes modules ---
from src.ui import UI
from src.data_acquisition.vosk_function import VoskRecognizer
from src.final_interaction.tts_piper import PiperTTS
from src.processing.chat import get_llm_response
from src.processing.function import choose_tool
from src.processing.server import LLMServerManager

# --- Import des utilitaires refactorisés ---
from src.utils import (
    obtenir_heure_formatee, 
    jouer_fichier_audio, 
    redimensionner_image_pour_ui, 
    analyser_image_via_api
)

# ==========================================
# 0. ARGUMENT PARSING & CONFIG
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--QT", action="store_true", help="Active le mode Robot QT (ROS)")
parser.add_argument("--pytest", action="store_true", help="Lance les tests unitaires")
args = parser.parse_args()

# Gestion Argument --pytest
if args.pytest:
    print("🧪 Lancement des tests (pytest -v)...")
    # On utilise sys.executable pour être sûr d'utiliser le même python
    result = subprocess.run([sys.executable, "-m", "pytest", "-v"])
    sys.exit(result.returncode)

IS_ROS_MODE = args.QT
ros_audio = None
ros_mic = None

# Imports conditionnels ROS
if IS_ROS_MODE:
    try:
        print("🤖 Mode QT activé : Chargement des modules ROS...")
        from src.ROS.PlayAudio import AudioController
        from src.ROS.ReadMicro import AudioStreamer
        # On ignore Display et Moove pour l'instant (focus Audio)
    except ImportError as e:
        print(f"❌ Erreur import ROS : {e}")
        print("Êtes-vous sûr d'avoir source votre environnement ROS ?")
        sys.exit(1)

# ==========================================
# 1. VARIABLES GLOBALES
# ==========================================
ui = None
config = None
vosk = None
tts = None
server_manager = None

# Gestion Caméra (Reste locale pour l'instant selon instructions)
webcam = None          
derniere_image = None  
arret_programme = False 

AUDIO_OUTPUT = "output.wav"
chat_history = [] 

# ==========================================
# 2. FONCTIONS UI & AUDIO
# ==========================================

def log(message):
    if ui:
        try:
            ui.after(0, lambda: ui.bottom_textbox.insert("end", message + "\n"))
            ui.after(0, lambda: ui.bottom_textbox.see("end"))
        except:
            pass
    print(f"[LOG] {message}")

def update_interface(box_id, text):
    target = None
    if box_id == 1: target = ui.textbox_1
    elif box_id == 2: target = ui.textbox_2
    elif box_id == 3: target = ui.textbox_3
    if target:
        ui.after(0, lambda: target.delete("1.0", "end"))
        ui.after(0, lambda: target.insert("end", text))

def parler(texte):
    """Génère l'audio et le joue (Local ou ROS)."""
    log(f"🔊 Je dis : {texte}")
    
    # 1. Génération du fichier WAV
    tts.synthesize(texte, AUDIO_OUTPUT, speaker_id=0)
    
    # 2. Lecture
    if IS_ROS_MODE and ros_audio:
        # Envoie le fichier généré au robot et le joue
        ros_audio.play(AUDIO_OUTPUT)
    else:
        # Lecture locale PC
        jouer_fichier_audio(AUDIO_OUTPUT)

# ==========================================
# 3. GESTION DE LA CAMÉRA
# ==========================================

def thread_camera():
    global webcam, derniere_image, arret_programme
    
    print("📸 Démarrage de la caméra (Locale)...")
    webcam = cv2.VideoCapture(0) 
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not webcam.isOpened():
        log("❌ Erreur: Impossible d'ouvrir la webcam.")
        return

    while not arret_programme:
        ret, frame = webcam.read()
        if ret:
            derniere_image = frame
            if ui:
                frame_resized = redimensionner_image_pour_ui(frame)
                if frame_resized is not None:
                    pil_img = Image.fromarray(frame_resized)
                    ui.after(0, lambda img=pil_img: ui.mettre_a_jour_image(img))
        time.sleep(0.04)

    webcam.release()
    print("📸 Caméra arrêtée.")

# ==========================================
# 4. ACTIONS SPÉCIFIQUES
# ==========================================

def action_donner_heure():
    texte = obtenir_heure_formatee()
    update_interface(3, texte)
    parler(texte)
    return texte

def action_voir():
    global derniere_image
    if derniere_image is None:
        return "Erreur: Caméra non disponible."

    log("👁️ Analyse LFM2-VL en cours...")
    url_vision = config['llm_server_vision']['url']
    description = analyser_image_via_api(derniere_image, url_vision)
    
    if description:
        log(f"👁️ Vision Brute : {description}")
        return description
    else:
        return "Je n'arrive pas à analyser l'image."

# ==========================================
# 5. LOGIQUE PRINCIPALE (ROUTEUR)
# ==========================================

def traiter_commande(user_text):
    global chat_history
    if not user_text.strip(): return
    
    log(f"👂 Entendu : {user_text}")
    update_interface(1, user_text)

    try:
        url_text = config['llm_server']['url']
        headers = config['llm_server']['headers']
        
        tool = choose_tool(user_text, url_text, headers)
        update_interface(2, f"Outil : {tool}")

        if tool == "get_time":
            response_text = action_donner_heure()
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "assistant", "content": response_text})

        elif tool == "get_vision":
            description_visuelle = action_voir()
            prompt_final = (
                f"CONTEXTE VISUEL : {description_visuelle}\n"
                f"DEMANDE UTILISATEUR : {user_text}\n"
                "Réponds lui naturellement."
            )
            chat_history.append({"role": "user", "content": prompt_final})
            response = get_llm_response(chat_history, url_text, headers)
            update_interface(3, response)
            parler(response)
            chat_history.append({"role": "assistant", "content": response})

        else:
            chat_history.append({"role": "user", "content": user_text})
            response = get_llm_response(chat_history, url_text, headers)
            update_interface(3, response)
            parler(response)
            chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        log(f"❌ Erreur Pipeline: {e}")

# ==========================================
# 6. GESTION THREADS & AUDIO
# ==========================================

def generateur_audio_ros():
    """Générateur qui pompe les chunks audio depuis le topic ROS."""
    if not ros_mic: return
    ros_mic.start_listening()
    while not arret_programme:
        chunk = ros_mic.get_audio_chunk()
        if chunk:
            yield chunk
        else:
            # Petite pause pour ne pas saturer le CPU si pas de data
            time.sleep(0.01)
    ros_mic.stop_listening()

def thread_ecoute():
    if IS_ROS_MODE:
        log("🎤 Micro ROS activé.")
        # On passe le générateur ROS à Vosk
        vosk.start_transcription(traiter_commande, audio_source_iterator=generateur_audio_ros)
    else:
        log("🎤 Micro Local activé.")
        # None = Utilisation PyAudio interne
        vosk.start_transcription(traiter_commande, audio_source_iterator=None)

# ==========================================
# 7. MAIN
# ==========================================

def fermeture_propre():
    global arret_programme
    print("Arrêt du système...")
    arret_programme = True 
    if server_manager: server_manager.stop()
    if ros_mic: ros_mic.stop_listening()
    if ui: ui.on_closing()
    sys.exit(0)

if __name__ == "__main__":
    try:
        with open("config/config.yaml", "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)

        server_manager = LLMServerManager()
        server_manager.start()

        # Initialisation ROS si demandée
        if IS_ROS_MODE:
            ros_audio = AudioController() # Pour parler (PlayAudio.py)
            ros_mic = AudioStreamer()     # Pour écouter (ReadMicro.py)

        ui = UI()
        ui.protocol("WM_DELETE_WINDOW", fermeture_propre)

        vosk = VoskRecognizer(model_path=config['models']['stt_vosk']['fr'])
        tts = PiperTTS(model_path=config['models']['tts_piper']['fr_upmc'])

        threading.Thread(target=thread_camera, daemon=True).start()
        threading.Thread(target=thread_ecoute, daemon=True).start()

        ui.mainloop()

    except KeyboardInterrupt:
        fermeture_propre()