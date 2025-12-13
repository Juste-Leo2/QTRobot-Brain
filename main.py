# main.py
import threading
import yaml
import sys
import cv2      
import time
import argparse
import subprocess
from PIL import Image

# --- FIX ENCODAGE WINDOWS ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# --- IMPORTS ---
from src.data_acquisition.vosk_function import VoskRecognizer
from src.final_interaction.tts_piper import PiperTTS
from src.processing.chat import get_llm_response
from src.processing.function import choose_tool # AGENT 1
from src.processing.agents import run_agent_3_gesture, run_agent_4_display # AGENTS 3 & 4
from src.processing.server import LLMServerManager
from src.data_acquisition.mtcnn_function import detect_faces, draw_faces
from src.utils import (obtenir_heure_formatee, jouer_fichier_audio, redimensionner_image_pour_ui, analyser_image_via_api)

# ==========================================
# 0. CONFIG & ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--QT", action="store_true", help="Mode QT Robot (Headless + ROS)")
parser.add_argument("--pytest", action="store_true", help="Lance les tests")
args = parser.parse_args()

IS_ROS_MODE = args.QT

if args.pytest:
    subprocess.run([sys.executable, "-m", "pytest", "-v"])
    sys.exit(0)

# --- ROS IMPORTS ---
ros_audio = None
ros_mic = None
ros_move = None
ros_display = None
ros_head = None

if IS_ROS_MODE:
    try:
        print("🤖 Chargement Modules ROS...")
        from src.ROS.PlayAudio import AudioController
        from src.ROS.ReadMicro import AudioStreamer
        from src.ROS.Moove import MoveController
        from src.ROS.Display import DisplayController
        from src.ROS.HeadControl import HeadController
    except ImportError as e:
        print(f"❌ Erreur ROS : {e}")
        sys.exit(1)
else:
    from src.ui import UI

# ==========================================
# 1. VARIABLES
# ==========================================
ui = None
config = None
vosk = None
tts = None
server_manager = None
webcam = None          
derniere_image = None  
arret_programme = False 
AUDIO_OUTPUT = "output.wav"
chat_history = [] 

# ==========================================
# 2. UI HELPERS
# ==========================================
def log(msg):
    print(f"[LOG] {msg}")
    if ui and not IS_ROS_MODE:
        try:
            ui.bottom_textbox.insert("end", msg + "\n")
            ui.bottom_textbox.see("end")
        except: pass

def update_ui_text(box_id, text):
    if IS_ROS_MODE: return # Pas d'UI graphique en mode QT
    target = {1: ui.textbox_1, 2: ui.textbox_2, 3: ui.textbox_3}.get(box_id)
    if target:
        ui.after(0, lambda: target.delete("1.0", "end"))
        ui.after(0, lambda: target.insert("end", text))

def parler(texte):
    log(f"🔊 {texte}")
    tts.synthesize(texte, AUDIO_OUTPUT, speaker_id=0)
    if IS_ROS_MODE and ros_audio:
        ros_audio.play(AUDIO_OUTPUT)
    else:
        jouer_fichier_audio(AUDIO_OUTPUT)

# ==========================================
# 3. CAMERA & TRACKING (OPTIMISÉ)
# ==========================================
def thread_camera():
    global webcam, derniere_image, arret_programme
    
    print("📸 Démarrage Caméra (Optimisation CPU activée)")
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not webcam.isOpened(): return

    # --- PARAMÈTRES D'OPTIMISATION ---
    FRAME_SKIP = 20  # Traiter 1 image sur 5 (Ajuste à 10 si ça lag encore)
    frame_count = 0
    current_faces = [] # Mémoire tampon des visages

    while not arret_programme:
        ret, frame = webcam.read()
        if ret:
            derniere_image = frame.copy()
            frame_count += 1
            
            # 1. DÉTECTION (Seulement toutes les N frames)
            if frame_count % FRAME_SKIP == 0:
                # Astuce : On réduit l'image pour MTCNN -> Beaucoup plus rapide
                # On divise la taille par 2 pour la détection (320x240)
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                
                detected_small = detect_faces(small_frame)
                
                # On remet les coordonnées à l'échelle (x2)
                current_faces = []
                for face in detected_small:
                    x, y, w, h = face['box']
                    face['box'] = [x*2, y*2, w*2, h*2] # Rescale
                    current_faces.append(face)

                # 2. COMMANDE MOTEUR (Uniquement quand on a une nouvelle détection)
                if current_faces and IS_ROS_MODE and ros_head:
                    # Plus gros visage
                    face = max(current_faces, key=lambda f: f['confidence'])
                    x, y, w, h = face['box']
                    cx, cy = x + w/2, y + h/2
                    h_img, w_img, _ = frame.shape
                    
                    # Commande moteur ROS
                    ros_head.move_head_to_center(cx, cy, w_img, h_img)

            # 3. DESSIN (Sur toutes les frames pour fluidité visuelle)
            # On utilise 'current_faces' qui garde la position d'il y a quelques millisecondes
            if current_faces:
                draw_faces(frame, current_faces)

            # 4. UI UPDATE (PC Local)
            if ui and not IS_ROS_MODE:
                res = redimensionner_image_pour_ui(frame)
                if res is not None:
                    ui.after(0, lambda i=Image.fromarray(res): ui.mettre_a_jour_image(i))
        
        # Petite pause pour ne pas saturer la boucle
        time.sleep(0.03) 
        
    webcam.release()

# ==========================================
# 4. PIPELINE PRINCIPAL
# ==========================================
def traiter_commande(user_text):
    global chat_history
    if not user_text.strip(): return
    
    update_ui_text(1, user_text)
    url = config['llm_server']['url']
    headers = config['llm_server']['headers']

    try:
        # --- AGENT 1 (OUTILS) ---
        tool = choose_tool(user_text, url, headers)
        update_ui_text(2, f"Outil: {tool}")
        log(f"Agent 1 a choisi : {tool}")

        response_text = ""

        # --- AGENT 2 (CHAT) ou FONCTION ---
        if tool == "get_time":
            response_text = obtenir_heure_formatee()
            
        elif tool == "get_vision":
            desc = action_voir()
            prompt_vis = f"Context: User showed an image described as '{desc}'. User said: '{user_text}'."
            chat_history.append({"role": "user", "content": prompt_vis})
            response_text = get_llm_response(chat_history, url, headers)
            
        else:
            chat_history.append({"role": "user", "content": user_text})
            response_text = get_llm_response(chat_history, url, headers)

        chat_history.append({"role": "assistant", "content": response_text})
        update_ui_text(3, response_text)

        # --- AGENT 3 (GESTUELLE) ---
        gesture_action = run_agent_3_gesture(user_text, response_text, config)
        if gesture_action:
            log(f"Agent 3 Action : {gesture_action['name']}")
            if IS_ROS_MODE and ros_move:
                if gesture_action['type'] == 'emotion':
                    ros_move.emotion(gesture_action['name'])
                else:
                    ros_move.gesture(gesture_action['name'])

        # --- AGENT 4 (DISPLAY) ---
        display_action = run_agent_4_display(user_text, response_text, config)
        if display_action:
            log(f"Agent 4 Display : [{display_action['type']}] {display_action['content']}")
            if IS_ROS_MODE and ros_display:
                if display_action['type'] == 'text':
                    ros_display.show_text(display_action['content'])
                else:
                    ros_display.show_image(display_action['content'])

        # Enfin, on parle
        parler(response_text)

    except Exception as e:
        log(f"Erreur Pipeline: {e}")

def action_voir():
    if derniere_image is None: return "Image noire."
    return analyser_image_via_api(derniere_image, config['llm_server_vision']['url']) or "Rien."

# ==========================================
# 5. AUDIO & MAIN LOOP
# ==========================================
def ros_audio_gen():
    if not ros_mic: return
    ros_mic.start_listening()
    while not arret_programme:
        chunk = ros_mic.get_audio_chunk()
        if chunk: yield chunk
        else: time.sleep(0.01)
    ros_mic.stop_listening()

def thread_ecoute():
    src = ros_audio_gen if IS_ROS_MODE else None
    vosk.start_transcription(traiter_commande, audio_source_iterator=src)

def shutdown():
    global arret_programme
    arret_programme = True
    if server_manager: server_manager.stop()
    if IS_ROS_MODE and ros_mic: ros_mic.stop_listening()
    if ui: ui.on_closing()
    sys.exit(0)

if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    server_manager = LLMServerManager()
    server_manager.start()

    if IS_ROS_MODE:
        ros_audio = AudioController()
        ros_mic = AudioStreamer()
        ros_move = MoveController()
        ros_display = DisplayController()
        ros_head = HeadController()
    else:
        ui = UI()
        ui.protocol("WM_DELETE_WINDOW", shutdown)

    vosk = VoskRecognizer(model_path=config['models']['stt_vosk']['fr'])
    tts = PiperTTS(model_path=config['models']['tts_piper']['fr_upmc'])

    threading.Thread(target=thread_camera, daemon=True).start()
    threading.Thread(target=thread_ecoute, daemon=True).start()

    if IS_ROS_MODE:
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: shutdown()
    else:
        ui.mainloop()