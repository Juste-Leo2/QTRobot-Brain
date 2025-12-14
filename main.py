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
from src.processing.server import LLMServerManager
from src.data_acquisition.mtcnn_function import detect_faces, draw_faces
from src.utils import (obtenir_heure_formatee, jouer_fichier_audio, redimensionner_image_pour_ui, analyser_image_via_api)

# ==========================================
# 0. CONFIG & ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--QT", action="store_true", help="Mode QT Robot (Headless + ROS)")
parser.add_argument("--API", type=str, help="Clé API Google GenAI", default=None)
parser.add_argument("--pytest", action="store_true", help="Lance les tests")
parser.add_argument("--no-moove", action="store_true", help="Désactive les mouvements...")

args = parser.parse_args()

IS_ROS_MODE = args.QT
API_KEY = args.API

if args.pytest:
    # --- MODIFICATION ICI ---
    cmd = [sys.executable, "-m", "pytest", "-v"]
    if API_KEY:
        # On passe la clé API à pytest via une option personnalisée
        cmd.append(f"--api-key={API_KEY}")
    
    subprocess.run(cmd)
    sys.exit(0)

# --- IMPORTS CONDITIONNELS (AGENTS) ---
if API_KEY:
    print(f"☁️ MODE API ACTIVÉ (Google Gemma 3)")
    from src.processing.api_google import GoogleGeminiHandler
else:
    print(f"🏠 MODE LOCAL (Llama/LFM)")
    from src.processing.chat import get_llm_response
    from src.processing.function import choose_tool
    from src.processing.agents import run_agent_3_gesture, run_agent_4_display

# --- ROS IMPORTS ---
ros_audio = None; ros_mic = None; ros_move = None; ros_display = None; ros_head = None

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
ui = None; config = None; vosk = None; tts = None; server_manager = None
webcam = None; derniere_image = None; arret_programme = False 
AUDIO_OUTPUT = "output.wav"
chat_history = [] 
api_handler = None 

# ==========================================
# 2. UI HELPERS
# ==========================================
def log(msg):
    print(f"[LOG] {msg}")
    if ui and not IS_ROS_MODE:
        try:
            ui.bottom_textbox.insert("end", msg + "\n"); ui.bottom_textbox.see("end")
        except: pass

def update_ui_text(box_id, text):
    if IS_ROS_MODE: return
    target = {1: ui.textbox_1, 2: ui.textbox_2, 3: ui.textbox_3}.get(box_id)
    if target:
        ui.after(0, lambda: target.delete("1.0", "end")); ui.after(0, lambda: target.insert("end", text))

def parler(texte):
    log(f"🔊 {texte}")
    tts.synthesize(texte, AUDIO_OUTPUT, speaker_id=0)
    if IS_ROS_MODE and ros_audio: ros_audio.play(AUDIO_OUTPUT)
    else: jouer_fichier_audio(AUDIO_OUTPUT)

# ==========================================
# 3. CAMERA & TRACKING (Optimisé)
# ==========================================
def thread_camera():
    global webcam, derniere_image, arret_programme
    print("📸 Démarrage Caméra...")
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640); webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not webcam.isOpened(): return
    
    FRAME_SKIP = 5; frame_count = 0; current_faces = []

    while not arret_programme:
        ret, frame = webcam.read()
        if ret:
            derniere_image = frame.copy()
            frame_count += 1
            
            # Detection optimisée
            if frame_count % FRAME_SKIP == 0:
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                det = detect_faces(small)
                current_faces = []
                for f in det:
                    x, y, w, h = f['box']
                    f['box'] = [x*2, y*2, w*2, h*2]
                    current_faces.append(f)
                
                # Commande ROS Tête
                if current_faces and IS_ROS_MODE and ros_head:
                    face = max(current_faces, key=lambda f: f['confidence'])
                    x, y, w, h = face['box']
                    ros_head.move_head_to_center(x + w/2, y + h/2, frame.shape[1], frame.shape[0])

            # Dessin
            if current_faces: draw_faces(frame, current_faces)
            
            # UI
            if ui and not IS_ROS_MODE:
                res = redimensionner_image_pour_ui(frame)
                if res is not None: 
                    ui.after(0, lambda i=Image.fromarray(res): ui.mettre_a_jour_image(i))
        time.sleep(0.03)
    webcam.release()

# ==========================================
# 4. PIPELINE DE TRAITEMENT
# ==========================================
def traiter_commande(user_text):
    global chat_history
    if not user_text.strip(): return
    update_ui_text(1, user_text)

    # Branche API
    if API_KEY and api_handler:
        traiter_commande_api(user_text)
        return

    # Branche Locale
    traiter_commande_local(user_text)

def traiter_commande_api(user_text):
    """Pipeline Google API"""
    try:
        tool = api_handler.router_api(user_text)
        update_ui_text(2, f"Outil (API): {tool}")
        log(f"☁️ Router: {tool}")

        tool_result = "None"
        if tool == "get_time": tool_result = obtenir_heure_formatee()
        elif tool == "get_vision":
            tool_result = f"Description visuelle: {action_voir()}"
            log(f"☁️ Vision Result: {tool_result}")

        fused = api_handler.generate_fused_response(user_text, chat_history, tool_result, config)
        
        response_text = fused["text"]
        action_robot = fused["action"]
        display_content = fused["display"]

        update_ui_text(3, response_text)
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": response_text})

        log(f"☁️ Action: {action_robot} | Display: {display_content}")
        
        if IS_ROS_MODE:
            if action_robot != "None" and ros_move:
                if action_robot in config['qt_robot']['emotions']: ros_move.emotion(action_robot)
                else: ros_move.gesture(action_robot)
            
            if display_content != "None" and ros_display:
                if "TEXT:" in display_content: ros_display.show_text(display_content.replace("TEXT:", ""))
                else: ros_display.show_image(display_content)

        parler(response_text)

    except Exception as e:
        log(f"❌ Erreur API Pipeline: {e}")

def traiter_commande_local(user_text):
    """Pipeline Locale (Restaurée avec Logs)"""
    try:
        url = config['llm_server']['url']
        headers = config['llm_server']['headers']
        
        # Agent 1
        tool = choose_tool(user_text, url, headers)
        update_ui_text(2, f"Outil: {tool}")
        log(f"Agent 1 a choisi : {tool}")
        
        response_text = ""
        # Agent 2 ou Fonction
        if tool == "get_time": response_text = obtenir_heure_formatee()
        elif tool == "get_vision":
            desc = action_voir()
            prompt = f"Vision: {desc}. User: {user_text}"
            chat_history.append({"role": "user", "content": prompt})
            response_text = get_llm_response(chat_history, url, headers)
        else:
            chat_history.append({"role": "user", "content": user_text})
            response_text = get_llm_response(chat_history, url, headers)

        chat_history.append({"role": "assistant", "content": response_text})
        update_ui_text(3, response_text)

        # --- MODIFICATION: CHECK ARGUMENT NO-MOOVE ---
        if not args.no_moove:
            # Agent 3 (Restauré)
            gesture = run_agent_3_gesture(user_text, response_text, config)
            if gesture:
                # LOG AFFICHÉ DANS TOUS LES CAS
                log(f"Agent 3 Action : {gesture['name']}")
                if IS_ROS_MODE and ros_move:
                    if gesture['type'] == 'emotion': ros_move.emotion(gesture['name'])
                    else: ros_move.gesture(gesture['name'])
                
            # Agent 4 (Restauré)
            display = run_agent_4_display(user_text, response_text, config)
            if display:
                # LOG AFFICHÉ DANS TOUS LES CAS
                log(f"Agent 4 Display : [{display['type']}] {display['content']}")
                if IS_ROS_MODE and ros_display:
                    if display['type'] == 'text': ros_display.show_text(display['content'])
                    else: ros_display.show_image(display['content'])
        else:
            log("🚫 Agents 3 (Geste) & 4 (Display) désactivés (--no-moove)")

        parler(response_text)
    except Exception as e:
        log(f"Erreur Local Pipeline: {e}")

def action_voir():
    if derniere_image is None: return "Rien."
    return analyser_image_via_api(derniere_image, config['llm_server_vision']['url']) or "Indéfini."

# ==========================================
# 5. BOUCLES & THREADS
# ==========================================

def ros_audio_gen():
    if not ros_mic: return
    ros_mic.start_listening()
    while not arret_programme:
        chunk = ros_mic.get_audio_chunk()
        yield chunk if chunk else time.sleep(0.01)
    ros_mic.stop_listening()

def thread_ecoute():
    src = ros_audio_gen if IS_ROS_MODE else None
    vosk.start_transcription(traiter_commande, audio_source_iterator=src)

def shutdown():
    global arret_programme; arret_programme = True
    if server_manager: server_manager.stop()
    if IS_ROS_MODE and ros_mic: ros_mic.stop_listening()
    if ui: ui.on_closing()
    sys.exit(0)

if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding='utf-8') as f: config = yaml.safe_load(f)

    if not API_KEY:
        server_manager = LLMServerManager()
        server_manager.start()
    else:
        api_handler = GoogleGeminiHandler(API_KEY)

    if IS_ROS_MODE:
        ros_audio = AudioController(); ros_mic = AudioStreamer()
        ros_move = MoveController(); ros_display = DisplayController(); ros_head = HeadController()
    else:
        ui = UI(); ui.protocol("WM_DELETE_WINDOW", shutdown)

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