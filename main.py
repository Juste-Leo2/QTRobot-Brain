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
import queue # <--- IMPORT CRITIQUE
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
from src.data_acquisition.mtcnn_function import detect_faces, draw_faces, select_priority_face
from src.data_acquisition.emotions import EmotionAnalyzer
from src.utils import (obtenir_heure_formatee, jouer_fichier_audio, redimensionner_image_pour_ui)

# ==========================================
# 0. CONFIG & ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--QT", action="store_true", help="Mode QT Robot (Active ROS + Bridge)")
parser.add_argument("--no-ui", action="store_true", help="Désactive l'interface graphique (Headless)")
parser.add_argument("--API", type=str, help="Clé API Google", default=None)
parser.add_argument("--OAPI", type=str, help="Clé API OpenRouter", default=None)
parser.add_argument("--pytest", action="store_true", help="Lance les tests")
parser.add_argument("--no-tools", action="store_true", help="Désactive les outils (Router) pour accélérer la réponse")
parser.add_argument("--name", type=str, default=None, help="Prénom du robot (Wake-word)")
parser.add_argument("--JKT", action="store_true", help="Active la connexion avec la veste connectée")

args = parser.parse_args()

IS_ROS_MODE = args.QT
SHOW_UI = not args.no_ui
API_KEY = args.API
OAPI_KEY = args.OAPI
ROBOT_NAME = args.name.lower() if args.name else None 

RPI_IP = "192.168.100.3"
RPI_USER = "qt"
RPI_PASS = "qtrobot"
RPI_SCRIPT = "/home/qt/Documents/inferenceQT0526.py"
RPI_VENV = "/home/qt/Documents/.venv/bin/activate"

if args.pytest:
    cmd = [sys.executable, "-m", "pytest", "-v"]
    if API_KEY: cmd.append(f"--api-key={API_KEY}")
    if OAPI_KEY: cmd.append(f"--oapi-key={OAPI_KEY}")
    subprocess.run(cmd)
    sys.exit(0)

if API_KEY:
    print(f"☁️ MODE API GOOGLE ACTIVÉ")
    from src.processing.api_google import GoogleGeminiHandler
elif OAPI_KEY:
    print(f"☁️ MODE API OPENROUTER ACTIVÉ")
    from src.processing.api_openrouter import OpenRouterHandler
else:
    print(f"🏠 MODE LOCAL")
    from src.processing.agent_chat import get_chat_response
    from src.processing.agent_fonction import choose_tool, execute_tool
    from src.processing.agent_animation import get_animation

ros_client = None
if IS_ROS_MODE:
    print("🤖 Mode ROS : Connexion Client...")
    from src.ROS.remote_client import RemoteRosClient
    ros_client = RemoteRosClient()
    ros_client.wakeup() 

if SHOW_UI:
    print("🖥️ Mode UI Desktop")
    from src.ui import UI
else:
    print("🚫 Mode Headless (Pas d'UI)")

jacket_manager = None
if args.JKT:
    print("🧥 Mode VESTE (JKT) Activé")
    try:
        from src.touch.robot_net import RaspberryManager
    except ImportError:
        print("❌ ERREUR: Impossible de trouver src.touch.robot_net")
        sys.exit(1)

# ==========================================
# 1. VARIABLES GLOBALES & QUEUE
# ==========================================
ui = None; config = None; vosk = None; tts_fr = None; tts_en = None; server_manager = None
current_lang = "fr"
webcam = None; derniere_image = None; arret_programme = False 
emotion_analyzer = None 
AUDIO_OUTPUT = "output.wav"
chat_history = [] 
api_handler = None 
IS_PROCESSING = False
current_emotion = "Neutre"
current_haptic_action = "Rien"

# --- NOUVELLE ARCHITECTURE ---
# File d'attente pour TOUTES les actions robot (First In, First Out)
ros_queue = queue.Queue()
# Lock pour l'accès bas niveau au socket (conflit Caméra vs Exécuteur)
robot_action_lock = threading.Lock()
# Timestamp de la dernière action LLM pour ignorer la veste
last_llm_action_ts = 0 
# Indique si l'exécuteur est en train de jouer un son (pour couper VOSK)
IS_ROBOT_SPEAKING = False 

# ==========================================
# 2. FONCTIONS UTILITAIRES
# ==========================================
def log(msg):
    print(f"[LOG] {msg}")
    if ui and SHOW_UI:
        try: 
            ui.bottom_textbox.insert("end", msg + "\n")
            ui.bottom_textbox.see("end")
        except: pass

def update_ui_text(box_id, text):
    if not SHOW_UI or not ui: return
    target = {1: ui.textbox_1, 2: ui.textbox_2, 3: ui.textbox_3}.get(box_id)
    if target:
        ui.after(0, lambda: target.delete("1.0", "end"))
        ui.after(0, lambda: target.insert("end", text))

def get_wav_duration(file_path):
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except: return 2.0

def generate_audio_only(texte):
    """Génère le fichier WAV sans le jouer."""
    global current_lang, tts_fr, tts_en
    print(f"🔊 [TTS Génération] {texte[:30]}...")
    try:
        tts_active = tts_en if current_lang == "en" else tts_fr
        tts_active.synthesize(texte, AUDIO_OUTPUT, speaker_id=0)
        return get_wav_duration(AUDIO_OUTPUT)
    except Exception as e:
        print(f"❌ Erreur TTS: {e}")
        return 0

# ==========================================
# 3. THREAD EXÉCUTEUR (Le Chef d'Orchestre)
# ==========================================
def thread_ros_executor():
    """
    Consomme les tâches de la file d'attente une par une.
    C'est le SEUL endroit qui envoie des commandes d'écriture au robot.
    """
    global IS_ROBOT_SPEAKING
    print("⚙️ [EXECUTOR] Démarré.")
    
    while not arret_programme:
        try:
            # On attend une tâche (bloquant, timeout 1s pour vérifier l'arrêt)
            task = ros_queue.get(timeout=1.0)
        except queue.Empty:
            continue
            
        type_action = task.get("type")
        data = task.get("data")
        
        # --- EXÉCUTION SÉCURISÉE ---
        with robot_action_lock:
            try:
                if IS_ROS_MODE and ros_client:
                    
                    if type_action == "gesture":
                        ros_client.gesture(data)
                        time.sleep(0.5) # Temps minimum pour le geste
                        
                    elif type_action == "emotion":
                        ros_client.emotion(data)
                        time.sleep(0.5) # Temps pour l'écran
                        
                    elif type_action == "text":
                        ros_client.show_text(data)
                        
                    elif type_action == "image":
                        ros_client.show_image(data)
                        
                    elif type_action == "audio":
                        # data est le chemin du fichier wav
                        import os
                        abs_path = os.path.abspath(data)
                        
                        IS_ROBOT_SPEAKING = True # On signale qu'on parle
                        ros_client.play(abs_path)
                        
                        # On attend la fin réelle de l'audio + marge
                        duration = task.get("duration", 2.0)
                        time.sleep(duration + 0.8)
                        
                        # Nettoyage buffer
                        ros_client.clear_socket_buffer()
                        IS_ROBOT_SPEAKING = False

                else:
                    # MODE PC LOCAL (Simulé)
                    if type_action == "audio":
                        IS_ROBOT_SPEAKING = True
                        jouer_fichier_audio(data)
                        IS_ROBOT_SPEAKING = False
                        
            except Exception as e:
                print(f"❌ [EXECUTOR] Erreur lors de l'exécution : {e}")
                IS_ROBOT_SPEAKING = False

        ros_queue.task_done()

# ==========================================
# 4. PIPELINE PRINCIPAL (Producteur 1)
# ==========================================
def traiter_commande(user_text):
    global chat_history, IS_PROCESSING, last_llm_action_ts
    if not user_text.strip(): return
    
    # --- WAKE WORD ---
    if ROBOT_NAME:
        text_clean = user_text.strip().lower()
        if not text_clean.startswith(ROBOT_NAME):
            print(f"🔇 Ignoré (Attendu: '{ROBOT_NAME}', Reçu: '{text_clean.split()[0]}...')")
            return
        user_text = user_text[len(ROBOT_NAME):].strip()
        if not user_text: return 
            
    print(f"\n🎤 [USER] \"{user_text}\"")
    IS_PROCESSING = True
    update_ui_text(1, user_text)

    try:
        # 1. LLM GENERATION
        if (API_KEY or OAPI_KEY) and api_handler:
            result = pipeline_api(user_text)
        else:
            result = pipeline_local(user_text)
            
        response_text = result["text"]
        action_robot = result["action"]
        display_content = result["display"]
        
        update_ui_text(3, response_text)
        log(f"Action: {action_robot} | Display: {display_content}")

        # 2. MISE EN FILE D'ATTENTE (QUEUE)
        # On enregistre le timestamp pour bloquer la veste pendant 10s
        last_llm_action_ts = time.time()
        
        # A. Ecran / Emotion
        if display_content != "None":
            if display_content.startswith("QT/"):
                ros_queue.put({"type": "emotion", "data": display_content})
            elif "TEXT:" in display_content:
                txt = display_content.replace("TEXT:", "")
                ros_queue.put({"type": "text", "data": txt})
            else:
                ros_queue.put({"type": "image", "data": display_content})

        # B. Geste
        if action_robot != "None":
            ros_queue.put({"type": "gesture", "data": action_robot})

        # C. Audio (On génère d'abord, puis on file l'ordre de jouer)
        duration = generate_audio_only(response_text)
        ros_queue.put({"type": "audio", "data": AUDIO_OUTPUT, "duration": duration})

    except Exception as e:
        log(f"❌ [ERREUR] {e}")
        IS_PROCESSING = False
    
    print("🟢 [PRET] Micro ouvert (Commandes en file d'attente)\n")
    IS_PROCESSING = False

def pipeline_api(user_text):
    if args.no_tools:
        tool = "None"
        update_ui_text(2, "Outil (API): Désactivé")
    else:
        tool = api_handler.router_api(user_text)
        update_ui_text(2, f"Outil (API): {tool}")
    
    tool_res = "None"
    if tool == "get_time": tool_res = obtenir_heure_formatee()
    
    fused = api_handler.generate_fused_response(user_text, chat_history, tool_res, config)
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": fused["text"]})
    return fused

def pipeline_local(user_text):
    global current_haptic_action
    url = config['llm_server']['url']
    
    tool = "None"
    if args.no_tools:
        update_ui_text(2, "Outil (Local): Désactivé")
    else:
        tool = choose_tool(user_text, url)
        update_ui_text(2, f"Outil (Local): {tool}")
        
    if tool != "None":
        res = execute_tool(tool)
        if res:
            if "switch_lang" in res:
                global current_lang
                current_lang = res["switch_lang"]
                print(f"🌍 [LANGUE] Changement de langue vers: {current_lang}")
                
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "assistant", "content": res["text"]})
            return res
            
    # 1. Pipeline Chat (Texte)
    print(f"🧠 [EMOTION] visuelle={current_emotion}, haptique={current_haptic_action}")
    resp_text = get_chat_response(chat_history, user_text, url,
                                   emotion_visuelle=current_emotion,
                                   action_haptique=current_haptic_action)
    current_haptic_action = "Rien"  # Reset après utilisation
    
    # 2. Pipeline Animation (JSON)
    anim_dict = get_animation(resp_text, url)
    act = anim_dict["action"]
    disp = anim_dict["display"]
    
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": resp_text})
    
    return {"text": resp_text, "action": act, "display": disp}



# ==========================================
# 5. THREADS (AUDIO INPUT, VESTE, CAMERA)
# ==========================================
def ros_audio_gen():
    """Ecoute audio ROS avec pause si le robot parle."""
    if not ros_client: return
    ros_client.start_listening()
    while not arret_programme:
        # Si le robot parle (via l'exécuteur) ou pense, on coupe l'oreille
        # On vérifie aussi si la queue n'est pas vide (action en attente)
        if IS_PROCESSING or IS_ROBOT_SPEAKING or not ros_queue.empty():
            try: ros_client.get_audio_chunk() # On vide le buffer pour éviter le larsen
            except: pass
            time.sleep(0.05)
            continue
    
        chunk = ros_client.get_audio_chunk()
        yield chunk if chunk else time.sleep(0.01)
    ros_client.stop_listening()

def should_pause_listening():
    return IS_PROCESSING or IS_ROBOT_SPEAKING or not ros_queue.empty()

def thread_ecoute():
    src = ros_audio_gen if IS_ROS_MODE else None
    print("🎤 Démarrage de l'écoute...")
    # Permet à vosk de lire la variable globale current_lang en temps réel
    lang_getter = lambda: current_lang
    vosk.start_transcription(traiter_commande, audio_source_iterator=src, pause_checker=should_pause_listening, language_getter=lang_getter)

def thread_jacket():
    """Thread Veste (Producteur 2)"""
    global chat_history, jacket_manager, current_haptic_action
    
    if not jacket_manager: return
    
    print("🧥 [JKT] Connexion au Raspberry Pi...")
    if not jacket_manager.connect_and_start():
        print("❌ [JKT] Echec connexion Veste.")
        return
        
    print("🧥 [JKT] Veste connectée et en écoute.")
    
    while not arret_programme:
        action = jacket_manager.get_data(clear_after=True)
        
        if action:
            # 1. CHECK COOLDOWN (10s après action LLM)
            time_since_llm = time.time() - last_llm_action_ts
            if time_since_llm < 10.0:
                print(f"🛡️ [JKT] Ignoré ({action}) - Priorité LLM active ({int(10-time_since_llm)}s restantes).")
                continue

            print(f"⚡ [JKT] Geste reçu : {action}")
            chat_history = []
            current_haptic_action = action
            
            phrase = ""
            geste = ""
            
            if action == "Tape":
                geste = "QT/happy"
                phrase = "Tu es incroyable !"
            elif action == "Frottement":
                geste = "QT/happy"
                phrase = "C'est doux."
            elif action == "Pincement":
                geste = "QT/angry"
                phrase = "Ça fait mal !"
            else:
                geste = "QT/confused"

            update_ui_text(3, f"Veste: {action}")

            # 2. MISE EN FILE D'ATTENTE (QUEUE)
            ros_queue.put({"type": "emotion", "data": geste})
            ros_queue.put({"type": "gesture", "data": geste})
            
            if phrase:
                # Génération locale TTS (pour ne pas bloquer l'executor)
                duration = generate_audio_only(phrase)
                # Envoi ordre lecture
                ros_queue.put({"type": "audio", "data": AUDIO_OUTPUT, "duration": duration})

        time.sleep(0.1)
    
    jacket_manager.stop()

def thread_camera():
    global webcam, derniere_image, current_emotion
    
    if not IS_ROS_MODE:
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            print("❌ Erreur: Webcam locale introuvable")
            return
            
    head_yaw = 0.0; head_pitch = 0.0 
    MAX_YAW = 80.0; MAX_PITCH_UP = -25.0; MAX_PITCH_DOWN = 25.0
    DEADZONE = 30; GAIN_X = 0.05; GAIN_Y = 0.05
    FREQ_TRACKING = 10; frame_count = 0

    last_emotion_label = None; last_box = None; last_face_center = None 

    while not arret_programme:
        frame = None
        if IS_ROS_MODE:
            if ros_client: frame = ros_client.get_camera_frame()
        else:
            ret, tmp_frame = webcam.read()
            if ret: frame = tmp_frame

        if frame is None:
            time.sleep(0.1); continue

        derniere_image = frame.copy()
        frame_count += 1
            
        if frame_count % FREQ_TRACKING == 0:
            small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            all_faces = detect_faces(small) 
            
            center_small = None
            if last_face_center:
                center_small = (last_face_center[0] * 0.5, last_face_center[1] * 0.5)

            priority_face, new_center_small = select_priority_face(all_faces, center_small)

            if priority_face:
                last_face_center = (new_center_small[0] * 2.0, new_center_small[1] * 2.0)
                
                if emotion_analyzer:
                    emo, _, box = emotion_analyzer.process_emotion(frame, [priority_face], scale_factor=2.0)
                    last_emotion_label = emo; last_box = box
                    if emo:
                        current_emotion = emo
                
                # Tracking Tête (Uniquement si queue vide pour éviter de saccader les gestes)
                if IS_ROS_MODE and ros_client and ros_queue.empty() and not IS_ROBOT_SPEAKING:
                    x, y, w, h = priority_face['box']
                    center_x = x + w / 2; center_y = y + h / 2
                    error_x = 160 - center_x; error_y = 120 - center_y 
                    move_needed = False

                    if abs(error_x) > DEADZONE:
                        head_yaw += error_x * GAIN_X; move_needed = True
                    if abs(error_y) > DEADZONE:
                        head_pitch -= error_y * GAIN_Y; move_needed = True

                    head_yaw = max(min(head_yaw, MAX_YAW), -MAX_YAW)
                    head_pitch = max(min(head_pitch, MAX_PITCH_DOWN), MAX_PITCH_UP)
                    
                    if move_needed:
                        # Protection Non-Bloquante : Si l'executor utilise le socket, on saute.
                        if robot_action_lock.acquire(blocking=False):
                            try:
                                ros_client.move_head(round(head_yaw, 1), round(head_pitch, 1))
                            finally:
                                robot_action_lock.release()

            else:
                last_emotion_label = None; last_box = None; last_face_center = None 

        # UI Dessin
        if last_box:
            x1, y1, x2, y2 = last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if last_emotion_label:
                cv2.putText(frame, last_emotion_label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if SHOW_UI and ui:
            res = redimensionner_image_pour_ui(frame)
            if res is not None:
                img_rgb = res 
                ui.after(0, lambda i=Image.fromarray(img_rgb): ui.mettre_a_jour_image(i))
                     
        time.sleep(0.04) 

    if not IS_ROS_MODE and webcam: webcam.release()

def shutdown():
    global arret_programme; arret_programme = True
    print("🛑 Arrêt du système...")
    if server_manager: server_manager.stop()
    if IS_ROS_MODE and ros_client: ros_client.stop_listening()
    if args.JKT and jacket_manager: jacket_manager.stop()
    if SHOW_UI and ui: ui.on_closing()
    sys.exit(0)

# ==========================================
# 6. MAIN
# ==========================================
if __name__ == "__main__":
    try:
        with open("config/config.yaml", "r", encoding='utf-8') as f: config = yaml.safe_load(f)
    except FileNotFoundError:
        print("⚠️ Config non trouvée, assurez-vous d'être à la racine.")

    if API_KEY:
        api_handler = GoogleGeminiHandler(API_KEY)
    elif OAPI_KEY:
        api_handler = OpenRouterHandler(OAPI_KEY)
    else:
        server_manager = LLMServerManager()
        server_manager.start()

    if args.JKT:
        jacket_manager = RaspberryManager(RPI_IP, RPI_USER, RPI_PASS, RPI_SCRIPT, RPI_VENV)
    
    if SHOW_UI:
        ui = UI(); ui.protocol("WM_DELETE_WINDOW", shutdown)
        
    models_stt_dict = {
        'fr': config['models']['stt_vosk']['fr'],
        'en': config['models']['stt_vosk']['en']
    }
    vosk = VoskRecognizer(models_dict=models_stt_dict)
    tts_fr = PiperTTS(model_path=config['models']['tts_piper']['fr_upmc'])
    tts_en = PiperTTS(model_path=config['models']['tts_piper']['en_amy'])
    
    emotion_analyzer = EmotionAnalyzer()

    # Lancement des threads
    threading.Thread(target=thread_ros_executor, daemon=True).start() # <-- LE CHEF D'ORCHESTRE
    threading.Thread(target=thread_camera, daemon=True).start()
    threading.Thread(target=thread_ecoute, daemon=True).start()
    
    if args.JKT:
        threading.Thread(target=thread_jacket, daemon=True).start()
    
    if SHOW_UI:
        print("🚀 SYSTEME PRET (Interface Active)")
        ui.mainloop()
    else:
        print("🚀 SYSTEME PRET (Mode Headless - Ctrl+C pour quitter)")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: shutdown()