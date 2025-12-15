import requests
import subprocess
import time
import sys
import os
import socket
from src.ROS.Transfer import FileTransfer

BRIDGE_URL = "http://127.0.0.1:5000"
AUDIO_PORT = 5001

class RemoteRosClient:
    def __init__(self):
        self.transfer = FileTransfer()
        self._ensure_server_running()
        self.audio_sock = None

    def _ensure_server_running(self):
        try:
            requests.get(f"{BRIDGE_URL}/status", timeout=1)
            print("✅ Pont ROS Connecté.")
            return
        except:
            print("⚠️ Démarrage du Pont ROS...")
        
        subprocess.run("fuser -k 5000/tcp", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("fuser -k 5001/tcp", shell=True, stderr=subprocess.DEVNULL)
        time.sleep(1)

        script_path = os.path.join(os.getcwd(), "src", "ROS", "bridge_server.py")
        subprocess.Popen(["/usr/bin/python3", script_path], stdout=sys.stdout, stderr=sys.stderr)

        for _ in range(10):
            try:
                requests.get(f"{BRIDGE_URL}/status", timeout=1)
                print("✅ Pont ROS Démarré !")
                return
            except: time.sleep(1)
        print("❌ ECHEC connexion ROS.")

    def _send(self, cmd, payload=""):
        try:
            requests.post(f"{BRIDGE_URL}/command", json={"command": cmd, "payload": payload}, timeout=1.0)
        except Exception as e:
            print(f"❌ Erreur envoi {cmd}: {e}")

    def wakeup(self): self._send("wakeup")
    def gesture(self, name): self._send("gesture", name)
    def emotion(self, name): self._send("emotion", name)
    def show_text(self, text): self._send("show_text", text)
    def move_head(self, yaw, pitch): self._send("head", f"{yaw},{pitch}")

    def show_image(self, filename):
        if filename.startswith("QT/"):
            self._send("show_image", filename)
        elif os.path.exists(filename):
            remote_path = self.transfer.send(filename, "stream_image")
            if remote_path: self._send("show_image", remote_path)

    def play(self, filename):
        if filename.startswith("QT/"):
            self._send("play", filename)
        elif os.path.exists(filename):
            remote_path = self.transfer.send(filename, "stream_audio")
            if remote_path: self._send("play", remote_path)

    # --- AUDIO MANAGEMENT ---
    def start_listening(self):
        try:
            self.audio_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.audio_sock.settimeout(2)
            self.audio_sock.connect(("127.0.0.1", AUDIO_PORT))
        except: self.audio_sock = None

    def get_audio_chunk(self):
        if not self.audio_sock: return None
        try: return self.audio_sock.recv(4096)
        except: return None

    def stop_listening(self):
        if self.audio_sock: 
            self.audio_sock.close(); self.audio_sock = None

    def clear_socket_buffer(self):
        """VIDE LE BUFFER AUDIO POUR EVITER L'ECHO"""
        if not self.audio_sock: return
        try:
            # On passe en mode non-bloquant pour lire tout ce qui traîne instantanément
            self.audio_sock.setblocking(0)
            while True:
                data = self.audio_sock.recv(4096)
                if not data: break
        except BlockingIOError:
            pass # Le buffer est vide, tout va bien
        except Exception as e:
            print(f"⚠️ Warning flush audio: {e}")
        finally:
            # On remet le mode bloquant avec timeout pour la suite
            self.audio_sock.setblocking(1)
            self.audio_sock.settimeout(2)