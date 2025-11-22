# main.py
import threading
import yaml
import sys
import cv2      
import time
from PIL import Image

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
# 1. VARIABLES GLOBALES
# ==========================================
ui = None
config = None
vosk = None
tts = None
server_manager = None

# Gestion Caméra
webcam = None          
derniere_image = None  
arret_programme = False 

AUDIO_OUTPUT = "output.wav"

# --- AJOUT : Historique de chat global ---
chat_history = [] 

# ==========================================
# 2. FONCTIONS UI
# ==========================================

def log(message):
    """Affiche un message dans la console de l'interface."""
    if ui:
        try:
            ui.after(0, lambda: ui.bottom_textbox.insert("end", message + "\n"))
            ui.after(0, lambda: ui.bottom_textbox.see("end"))
        except:
            pass
    print(f"[LOG] {message}")

def update_interface(box_id, text):
    """Met à jour une des boites de texte."""
    target = None
    if box_id == 1: target = ui.textbox_1
    elif box_id == 2: target = ui.textbox_2
    elif box_id == 3: target = ui.textbox_3
    if target:
        ui.after(0, lambda: target.delete("1.0", "end"))
        ui.after(0, lambda: target.insert("end", text))

def parler(texte):
    """Génère l'audio et le joue via utilitaire."""
    log(f"🔊 Je dis : {texte}")
    tts.synthesize(texte, AUDIO_OUTPUT, speaker_id=0)
    jouer_fichier_audio(AUDIO_OUTPUT)

# ==========================================
# 3. GESTION DE LA CAMÉRA
# ==========================================

def thread_camera():
    global webcam, derniere_image, arret_programme
    
    print("📸 Démarrage de la caméra...")
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
    global chat_history  # On accède à l'historique global
    
    if not user_text.strip(): return
    
    log(f"👂 Entendu : {user_text}")
    update_interface(1, user_text)

    try:
        url_text = config['llm_server']['url']
        headers = config['llm_server']['headers']
        
        # 1. Choix de l'outil
        tool = choose_tool(user_text, url_text, headers)
        update_interface(2, f"Outil : {tool}")

        # 2. Exécution selon l'outil
        if tool == "get_time":
            response_text = action_donner_heure()
            
            # On ajoute l'interaction à l'historique pour que l'IA s'en souvienne
            # (ex: si après on dit "Merci", elle sait pourquoi)
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "assistant", "content": response_text})

        elif tool == "get_vision":
            description_visuelle = action_voir()
            
            # On construit un prompt contextuel pour l'IA
            prompt_final = (
                f"CONTEXTE VISUEL (Ce que voient tes yeux) : {description_visuelle}\n"
                f"DEMANDE UTILISATEUR : {user_text}\n"
                "Réponds lui naturellement en prenant en compte le contexte visuel."
            )
            
            # On ajoute ce prompt enrichi à l'historique
            chat_history.append({"role": "user", "content": prompt_final})
            
            # On envoie tout l'historique (chat.py gérera la fenêtre glissante)
            response = get_llm_response(chat_history, url_text, headers)
            
            # On met à jour l'UI et l'audio
            update_interface(3, response)
            parler(response)
            
            # On sauvegarde la réponse de l'IA dans l'historique
            chat_history.append({"role": "assistant", "content": response})

        else:
            # CAS CLASSIQUE (Conversation normale)
            chat_history.append({"role": "user", "content": user_text})
            
            # Appel LLM avec tout l'historique
            response = get_llm_response(chat_history, url_text, headers)
            
            update_interface(3, response)
            parler(response)
            
            # Sauvegarde réponse
            chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        log(f"❌ Erreur Pipeline: {e}")

def thread_ecoute():
    log("🎤 Micro activé.")
    vosk.start_transcription(traiter_commande)

# ==========================================
# 6. MAIN
# ==========================================

def fermeture_propre():
    global arret_programme
    print("Arrêt du système...")
    arret_programme = True 
    if server_manager: server_manager.stop()
    if ui: ui.on_closing()
    sys.exit(0)

if __name__ == "__main__":
    try:
        with open("config/config.yaml", "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)

        server_manager = LLMServerManager()
        server_manager.start()

        ui = UI()
        ui.protocol("WM_DELETE_WINDOW", fermeture_propre)

        vosk = VoskRecognizer(model_path=config['models']['stt_vosk']['fr'])
        tts = PiperTTS(model_path=config['models']['tts_piper']['fr_upmc'])

        threading.Thread(target=thread_camera, daemon=True).start()
        threading.Thread(target=thread_ecoute, daemon=True).start()

        ui.mainloop()

    except KeyboardInterrupt:
        fermeture_propre()