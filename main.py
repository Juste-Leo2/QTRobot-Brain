# main.py
import threading
import yaml
import sys
import cv2      
import time
import argparse
import subprocess
import wave
import contextlib
from PIL import Image
import numpy as np

# --- FIX ENCODAGE ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# --- IMPORTS ---
from src.data_acquisition.vosk_function import VoskRecognizer
from src.final_interaction.tts_piper import PiperTTS
from src.processing.server import LLMServerManager
from src.data_acquisition.mtcnn_function import detect_faces, draw_faces
from src.utils import (obtenir_heure_formatee, jouer_fichier_audio, redimensionner_image_pour_ui, analyser_image_via_api)

# ==========================================
# 0. CONFIG & ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--QT", action="store_true", help="Mode QT Robot (Active ROS + Bridge)")
parser.add_argument("--no-ui", action="store_true", help="Désactive l'interface graphique (Headless)")
parser.add_argument("--API", type=str, help="Clé API Google", default=None)
parser.add_argument("--pytest", action="store_true", help="Lance les tests")
parser.add_argument("--no-moove", action="store_true", help="Désactive les mouvements")
args = parser.parse_args()

IS_ROS_MODE = args.QT
SHOW_UI = not args.no_ui
API_KEY = args.API

if args.pytest:
    cmd = [sys.executable, "-m", "pytest", "-v"]
    if API_KEY: cmd.append(f"--api-key={API_KEY}")
    subprocess.run(cmd)
    sys.exit(0)

# Sélection du Backend LLM
if API_KEY:
    print(f"☁️ MODE API ACTIVÉ")
    from src.processing.api_google import GoogleGeminiHandler
else:
    print(f"🏠 MODE LOCAL")
    from src.processing.chat import get_llm_response
    from src.processing.function import choose_tool
    from src.processing.agents import run_agent_3_gesture, run_agent_4_display

# Sélection du Frontend (ROS ou UI)
ros_client = None
if IS_ROS_MODE:
    print("🤖 Mode ROS : Connexion Client...")
    from src.ROS.remote_client import RemoteRosClient
    ros_client = RemoteRosClient()
    ros_client.wakeup() # Premier réveil

if SHOW_UI:
    print("🖥️ Mode UI Desktop")
    from src.ui import UI
else:
    print("🚫 Mode Headless (Pas d'UI)")

# ==========================================
# 1. VARIABLES GLOBALES
# ==========================================
ui = None; config = None; vosk = None; tts = None; server_manager = None
webcam = None; derniere_image = None; arret_programme = False 
AUDIO_OUTPUT = "output.wav"
chat_history = [] 
api_handler = None 
IS_PROCESSING = False 

# ==========================================
# 2. FONCTIONS UTILITAIRES
# ==========================================
def log(msg):
    """Affiche les logs dans la console et dans l'UI si disponible"""
    print(f"[LOG] {msg}")
    if ui and SHOW_UI:
        try: ui.bottom_textbox.insert("end", msg + "\n"); ui.bottom_textbox.see("end")
        except: pass

def update_ui_text(box_id, text):
    """Met à jour les zones de texte de l'UI de manière thread-safe"""
    if not SHOW_UI or not ui: return
    target = {1: ui.textbox_1, 2: ui.textbox_2, 3: ui.textbox_3}.get(box_id)
    if target:
        ui.after(0, lambda: target.delete("1.0", "end")); ui.after(0, lambda: target.insert("end", text))

def get_wav_duration(file_path):
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes(); rate = f.getframerate()
            return frames / float(rate)
    except: return 2.0

def parler(texte):
    """Synthétise et joue l'audio (ROS ou Local)"""
    print(f"🔊 [TTS] {texte[:50]}...")
    tts.synthesize(texte, AUDIO_OUTPUT, speaker_id=0)
    duration = get_wav_duration(AUDIO_OUTPUT)
    
    if IS_ROS_MODE and ros_client:
        ros_client.play(AUDIO_OUTPUT)
    else:
        # Mode UI / PC Local
        jouer_fichier_audio(AUDIO_OUTPUT)
        
    return duration

# ==========================================
# 3. PIPELINE PRINCIPAL
# ==========================================
def traiter_commande(user_text):
    global chat_history, IS_PROCESSING
    if not user_text.strip(): return
    
    print(f"\n🎤 [USER] \"{user_text}\"")
    IS_PROCESSING = True
    update_ui_text(1, user_text)

    try:
        # --- GENERATION LLM ---
        if API_KEY and api_handler:
            result = pipeline_api(user_text)
        else:
            result = pipeline_local(user_text)
            
        response_text = result["text"]
        action_robot = result["action"]
        display_content = result["display"]
        
        # UI Update
        update_ui_text(3, response_text)
        log(f"Action: {action_robot} | Display: {display_content}")

        # --- ACTIONS PHYSIQUES (Seulement si ROS) ---
        if IS_ROS_MODE and ros_client:
            
            # 1. Ecran / Emotion
            if display_content != "None":
                if display_content.startswith("QT/"):
                    ros_client.emotion(display_content)
                elif "TEXT:" in display_content:
                    ros_client.show_text(display_content.replace("TEXT:", ""))
                else:
                    ros_client.show_image(display_content)
                time.sleep(1.0) 

            # 2. Mouvement
            if action_robot != "None" and not args.no_moove:
                ros_client.gesture(action_robot)
                time.sleep(0.5)

            # 3. Vidage buffer audio
            ros_client.clear_socket_buffer()

        # --- AUDIO (Pour TOUS les modes) ---
        audio_duration = parler(response_text)
        
        # Attente pour ne pas écouter sa propre voix
        if IS_ROS_MODE:
            time.sleep(audio_duration + 0.5)
            ros_client.clear_socket_buffer() 

    except Exception as e:
        log(f"❌ [ERREUR] {e}")
        IS_PROCESSING = False
    
    print("🟢 [PRET] Micro ouvert\n")
    IS_PROCESSING = False

def pipeline_api(user_text):
    """Gestion via API Google"""
    tool = api_handler.router_api(user_text)
    update_ui_text(2, f"Outil (API): {tool}")
    
    tool_res = "None"
    if tool == "get_time": tool_res = obtenir_heure_formatee()
    elif tool == "get_vision": tool_res = f"Vision: {action_voir()}"
    
    fused = api_handler.generate_fused_response(user_text, chat_history, tool_res, config)
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": fused["text"]})
    return fused

def pipeline_local(user_text):
    """Gestion via Serveur Local"""
    url = config['llm_server']['url']; headers = config['llm_server']['headers']
    
    tool = choose_tool(user_text, url, headers)
    update_ui_text(2, f"Outil (Local): {tool}")
    
    resp_text = ""
    if tool == "get_time": resp_text = obtenir_heure_formatee()
    elif tool == "get_vision":
        prompt = f"Vision: {action_voir()}. User: {user_text}"
        chat_history.append({"role": "user", "content": prompt})
        resp_text = get_llm_response(chat_history, url, headers)
    else:
        chat_history.append({"role": "user", "content": user_text})
        resp_text = get_llm_response(chat_history, url, headers)
    
    # Agents Gesture/Display (Local seulement pour la decision, execution ROS plus haut)
    act = "None"; disp = "None"
    if not args.no_moove:
        g = run_agent_3_gesture(user_text, resp_text, config)
        if g: act = g['name']
        d = run_agent_4_display(user_text, resp_text, config)
        if d: disp = d['content'] if d['type'] == 'image' else f"TEXT:{d['content']}"
    
    chat_history.append({"role": "assistant", "content": resp_text})
    return {"text": resp_text, "action": act, "display": disp}

def action_voir():
    if derniere_image is None: return "Rien (Caméra éteinte ou noire)."
    return analyser_image_via_api(derniere_image, config['llm_server_vision']['url']) or "Indéfini."

# ==========================================
# 4. THREADS (CAMERA & AUDIO)
# ==========================================
def ros_audio_gen():
    """Générateur audio pour ROS"""
    if not ros_client: return
    ros_client.start_listening()
    while not arret_programme:
        if IS_PROCESSING:
            try: ros_client.get_audio_chunk()
            except: pass
            time.sleep(0.05)
            continue
        chunk = ros_client.get_audio_chunk()
        yield chunk if chunk else time.sleep(0.01)
    ros_client.stop_listening()

def thread_ecoute():
    """Thread principal d'écoute (VOSK)"""
    src = ros_audio_gen if IS_ROS_MODE else None
    print("🎤 Démarrage de l'écoute...")
    vosk.start_transcription(traiter_commande, audio_source_iterator=src)

def thread_camera():
    """Thread Caméra : Tracking Visage corrigé en DEGRÉS"""
    global webcam, derniere_image
    
    # Init Webcam locale si pas ROS
    if not IS_ROS_MODE:
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            print("❌ Erreur: Webcam locale introuvable")
            return
            
    # --- VARIABLES DE TRACKING (En Degrés maintenant) ---
    head_yaw = 0.0   
    head_pitch = 0.0 
    
    # PARAMETRES EN DEGRES
    # Si le robot va de -90 (droite) à +90 (gauche)
    MAX_YAW = 80.0       
    # Le pitch est souvent plus limité (-30 à +30)
    MAX_PITCH_UP = -25.0
    MAX_PITCH_DOWN = 25.0
    
    # Zone morte (en pixels)
    DEADZONE = 30 
    
    # Vitesse de réaction (Gain)
    GAIN_X = 0.05
    GAIN_Y = 0.05

    FREQ_TRACKING = 10  
    frame_count = 0

    while not arret_programme:
        frame = None
        
        # Acquisition
        if IS_ROS_MODE:
            if ros_client: frame = ros_client.get_camera_frame()
        else:
            ret, tmp_frame = webcam.read()
            if ret: frame = tmp_frame

        if frame is None:
            time.sleep(0.1)
            continue

        derniere_image = frame.copy()
        frame_count += 1
            
        # TRACKING
        if IS_ROS_MODE and ros_client and not args.no_moove and not IS_PROCESSING:
            
            if frame_count % FREQ_TRACKING == 0:
                small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                det = detect_faces(small)
                
                if det:
                    f = max(det, key=lambda x:x['confidence'])
                    x, y, w, h = f['box']
                    
                    # Centre (320x240) -> 160, 120
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    error_x = 160 - center_x
                    error_y = 120 - center_y 
                    
                    move_needed = False

                    # -- YAW (Degrés) --
                    if abs(error_x) > DEADZONE:
                        head_yaw += error_x * GAIN_X
                        move_needed = True
                    
                    # -- PITCH (Degrés) --
                    if abs(error_y) > DEADZONE:
                        head_pitch -= error_y * GAIN_Y 
                        move_needed = True

                    # -- LIMITES (Clamping en degrés) --
                    head_yaw = max(min(head_yaw, MAX_YAW), -MAX_YAW)
                    head_pitch = max(min(head_pitch, MAX_PITCH_DOWN), MAX_PITCH_UP)

                    if move_needed:
                        # On envoie des entiers ou floats simples (ex: 25.5, -10.0)
                        ros_client.move_head(round(head_yaw, 1), round(head_pitch, 1))

        # UI Update
        if SHOW_UI and ui:
            res = redimensionner_image_pour_ui(frame)
            if res is not None:
                img_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                ui.after(0, lambda i=Image.fromarray(img_rgb): ui.mettre_a_jour_image(i))
                    
        time.sleep(0.04) 

    if not IS_ROS_MODE and webcam:
        webcam.release()


def shutdown():
    global arret_programme; arret_programme = True
    print("🛑 Arrêt du système...")
    if server_manager: server_manager.stop()
    if IS_ROS_MODE and ros_client: ros_client.stop_listening()
    if SHOW_UI and ui: ui.on_closing()
    sys.exit(0)

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    # Chargement Config
    try:
        with open("config/config.yaml", "r", encoding='utf-8') as f: config = yaml.safe_load(f)
    except FileNotFoundError:
        print("⚠️ Config non trouvée, assurez-vous d'être à la racine.")

    # Gestionnaire Serveur (si pas d'API Key)
    if not API_KEY:
        server_manager = LLMServerManager()
        server_manager.start()
    else:
        api_handler = GoogleGeminiHandler(API_KEY)
    
    # Initialisation Interface
    if SHOW_UI:
        ui = UI(); ui.protocol("WM_DELETE_WINDOW", shutdown)
        
    # Initialisation Modèles
    vosk = VoskRecognizer(model_path=config['models']['stt_vosk']['fr'])
    tts = PiperTTS(model_path=config['models']['tts_piper']['fr_upmc'])
    
    # Lancement Threads
    threading.Thread(target=thread_camera, daemon=True).start()
    threading.Thread(target=thread_ecoute, daemon=True).start()
    
    if SHOW_UI:
        print("🚀 SYSTEME PRET (Interface Active)")
        ui.mainloop()
    else:
        print("🚀 SYSTEME PRET (Mode Headless - Ctrl+C pour quitter)")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: shutdown()