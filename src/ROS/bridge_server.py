#!/usr/bin/env python3
import sys
import os
import socket
import threading
import time
import subprocess
import cv2
import numpy as np
from flask import Flask, request, jsonify, Response

# --- PATH ROS ---
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path not in sys.path: sys.path.insert(0, ros_path)

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
# On n'a plus besoin d'importer Float64MultiArray car on passe par le terminal

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ReadMicro import AudioStreamer
except ImportError:
    print("❌ ReadMicro.py introuvable")
    sys.exit(1)

app = Flask(__name__)
micro_stream = None
video_stream = None

# --- FONCTIONS UTILITAIRES ---

def run_rostopic_blocking(topic, msg_type, content):
    """Pour les gestes et émotions (Commandes ponctuelles)"""
    # -1 = une seule fois, --latch = garde l'info
    cmd = ["rostopic", "pub", "-1", "--latch", topic, msg_type, f"data: '{content}'"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_rostopic_non_blocking(topic, msg_type, content):
    """Pour la TETE : On lance et on n'attend pas (Fire and Forget)"""
    # On construit la commande string complète
    cmd_str = f"rostopic pub -1 {topic} {msg_type} \"data: {content}\""
    # Popen lance le processus en parallèle -> Pas de lag vidéo
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def play_audio_service(full_path):
    directory = os.path.dirname(full_path) + "/"
    filename = os.path.basename(full_path)
    args = f"{{filename: '{filename}', filepath: '{directory}'}}"
    cmd = f"rosservice call /qt_robot/audio/play \"{args}\""
    subprocess.Popen(cmd, shell=True)

# --- CLASS VIDEO STREAMER ---
class VideoStreamer:
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_frame = None
        self.lock = threading.Lock()
        self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        print("📷 Video Streamer abonné à /camera/color/image_raw")

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            with self.lock:
                self.latest_frame = cv_image
        except CvBridgeError as e: pass

    def get_jpeg(self):
        with self.lock:
            if self.latest_frame is None: return None
            ret, buffer = cv2.imencode('.jpg', self.latest_frame)
            if ret: return buffer.tobytes()
            return None

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

@app.route('/camera', methods=['GET'])
def get_camera_frame():
    if video_stream:
        frame_data = video_stream.get_jpeg()
        if frame_data: return Response(frame_data, mimetype='image/jpeg')
    return Response(status=204)

@app.route('/command', methods=['POST'])
def command():
    data = request.json
    cmd = data.get('command')
    payload = data.get('payload')
    
    try:
        if cmd == "wakeup":
            # Wakeup important : on attend qu'il finisse
            subprocess.run(["rosservice", "call", "/qt_robot/motors/home", "[]"], stdout=subprocess.DEVNULL)
            
        elif cmd == "gesture":
            run_rostopic_blocking("/qt_robot/gesture/play", "std_msgs/String", payload)
            
        elif cmd == "emotion":
            run_rostopic_blocking("/qt_robot/emotion/show", "std_msgs/String", payload)
            
        elif cmd == "head":
            # --- RETOUR A LA METHODE ROSTOPIC (SHELL) ---
            # Payload reçu : "0.5,0.2"
            # Format attendu par rostopic : [0.5, 0.2]
            vals = f"[{payload}]"
            
            # Utilisation de Popen (Non bloquant)
            run_rostopic_non_blocking(
                "/qt_robot/head_position/command", 
                "std_msgs/Float64MultiArray", 
                vals
            )
            # print(f"Move Head: {vals}")

        elif cmd == "play":
            if str(payload).startswith("QT/"):
                run_rostopic_blocking("/qt_robot/audio/play", "std_msgs/String", payload)
            else:
                play_audio_service(payload)

        elif cmd == "show_text":
            run_rostopic_blocking("/qt_robot/screen/show_text", "std_msgs/String", payload)

        elif cmd == "show_image":
            run_rostopic_blocking("/qt_robot/screen/show_image", "std_msgs/String", payload)

        return jsonify({"res": "ok"})
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    rospy.init_node('qt_bridge_shell', anonymous=True, disable_signals=True)
    
    micro_stream = AudioStreamer()
    video_stream = VideoStreamer() 
    
    threading.Thread(target=audio_server, daemon=True).start()
    print("🚀 BRIDGE SERVER: Mode Shell (Robust & Compatible).")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)