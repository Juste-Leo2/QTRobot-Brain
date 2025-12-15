#!/usr/bin/env python3
import sys
import os
import socket
import threading
import time
import subprocess
from flask import Flask, request, jsonify

# --- PATH ROS ---
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path not in sys.path: sys.path.insert(0, ros_path)

import rospy
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ReadMicro import AudioStreamer
except ImportError:
    print("❌ ReadMicro.py introuvable")
    sys.exit(1)

app = Flask(__name__)
micro_stream = None

# --- FONCTIONS ---

def send_rostopic_blocking(topic, msg_type, content):
    """Envoie une commande ROS et attend qu'elle soit partie (fiabilité)"""
    # -1 : publie une fois
    # --latch : garde la dernière valeur active pour les nouveaux abonnés (fixe l'émotion qui disparait)
    cmd = [
        "rostopic", "pub", "-1", "--latch", topic, msg_type, f"data: '{content}'"
    ]
    # On utilise run() au lieu de Popen() pour bloquer le thread 0.1s le temps que ça parte
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def play_audio_service(full_path):
    directory = os.path.dirname(full_path) + "/"
    filename = os.path.basename(full_path)
    args = f"{{filename: '{filename}', filepath: '{directory}'}}"
    cmd = f"rosservice call /qt_robot/audio/play \"{args}\""
    print(f"🔊 SERVICE: {cmd}")
    subprocess.Popen(cmd, shell=True)

# --- SERVEUR AUDIO ---
def audio_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('0.0.0.0', 5001))
        server.listen(1)
        while not rospy.is_shutdown():
            conn, _ = server.accept()
            if not micro_stream.is_listening: micro_stream.start_listening()
            while True:
                chunk = micro_stream.get_audio_chunk()
                if chunk:
                    try: conn.sendall(chunk)
                    except: break
                else: time.sleep(0.005)
            conn.close()
            micro_stream.stop_listening()
    except: pass

# --- ROUTES ---
@app.route('/status', methods=['GET'])
def status(): return jsonify({"status": "ready"})

@app.route('/command', methods=['POST'])
def command():
    data = request.json
    cmd = data.get('command')
    payload = data.get('payload')
    
    print(f"📥 RECU: {cmd} -> {payload}")
    
    try:
        if cmd == "wakeup":
            # Appel bloquant pour être sûr que les moteurs s'activent
            subprocess.run(["rosservice", "call", "/qt_robot/motors/home", "[]"], stdout=subprocess.DEVNULL)
            
        elif cmd == "gesture":
            # BLOCKING CALL : Règle le problème du robot qui ne bouge pas
            send_rostopic_blocking("/qt_robot/gesture/play", "std_msgs/String", payload)
            
        elif cmd == "emotion":
            # BLOCKING CALL + LATCH : Règle le problème de l'émotion qui disparait
            send_rostopic_blocking("/qt_robot/emotion/show", "std_msgs/String", payload)
            
        elif cmd == "head":
            vals = f"[{payload}]"
            # Ici on laisse Popen pour la fluidité du tracking
            cmd_str = f"rostopic pub -1 /qt_robot/head_position/command std_msgs/Float64MultiArray \"data: {vals}\""
            subprocess.Popen(cmd_str, shell=True)

        elif cmd == "play":
            if str(payload).startswith("QT/"):
                send_rostopic_blocking("/qt_robot/audio/play", "std_msgs/String", payload)
            else:
                play_audio_service(payload)

        elif cmd == "show_text":
            send_rostopic_blocking("/qt_robot/screen/show_text", "std_msgs/String", payload)

        elif cmd == "show_image":
            send_rostopic_blocking("/qt_robot/screen/show_image", "std_msgs/String", payload)

        return jsonify({"res": "ok"})
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    rospy.init_node('qt_bridge_shell', anonymous=True, disable_signals=True)
    micro_stream = AudioStreamer()
    threading.Thread(target=audio_server, daemon=True).start()
    print("🚀 BRIDGE SERVER (Robust Mode).")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)