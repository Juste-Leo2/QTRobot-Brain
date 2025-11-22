import subprocess
import time
import requests
import yaml
import os
import signal
from pathlib import Path

class LLMServerManager:
    """
    Gère le cycle de vie des DEUX serveurs (Texte et Vision).
    """
    def __init__(self, config_path="config/config.yaml"):
        self.process_text = None
        self.process_vision = None
        self.config = self._load_config(config_path)
        
    def _load_config(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return yaml.safe_load(f)

    def start(self):
        """Lance les deux serveurs."""
        print("🚀 Initialisation des serveurs IA...")
        
        # 1. Démarrage Serveur TEXTE
        ok_text = self._start_server_text()
        
        # 2. Démarrage Serveur VISION
        ok_vision = self._start_server_vision()

        return ok_text and ok_vision

    def _start_server_text(self):
        """Logique pour lancer le serveur textuel (Port 8084)"""
        print("   -> Démarrage Llama Texte (8084)...")
        
        exe_path = Path(self.config['executables']['llama_server']['path'])
        model_path = Path(self.config['models']['llm']['lfm_8b'])
        args_template = self.config['executables']['llama_server']['args'] # "-m {model_path}..."
        
        # On remplit les trous dans la commande
        cmd_args = args_template.format(model_path=str(model_path))
        full_cmd = [str(exe_path)] + cmd_args.split()

        self.process_text = self._launch_process(full_cmd)
        return self._wait_for_url(self.config['llm_server']['url'])

    def _start_server_vision(self):
        """Logique pour lancer le serveur vision (Port 8088)"""
        print("   -> Démarrage Llama Vision (8088)...")

        exe_path = Path(self.config['executables']['llama_server_vision']['path'])
        model_path = Path(self.config['models']['llm']['LFM2-VL-450M-Q4'])
        mmproj_path = Path(self.config['models']['llm']['mmproj-LFM2-VL-450M-Q8'])
        args_template = self.config['executables']['llama_server_vision']['args']

        # Ici on a deux variables à remplir : le modèle et le projecteur (mmproj)
        cmd_args = args_template.format(
            model_path=str(model_path),
            model_path_mmproj=str(mmproj_path)
        )
        full_cmd = [str(exe_path)] + cmd_args.split()

        self.process_vision = self._launch_process(full_cmd)
        base_url = "http://localhost:8088/health" 
        return self._wait_for_url(base_url)

    def _launch_process(self, command):
        """Fonction utilitaire pour lancer un .exe sans fenêtre noire."""
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        try:
            return subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags
            )
        except Exception as e:
            print(f"❌ Erreur lancement EXE : {e}")
            return None

    def _wait_for_url(self, url, timeout=30):
        """Ping une URL jusqu'à ce qu'elle réponde."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                requests.get(url, timeout=1)
                return True
            except requests.exceptions.RequestException:
                time.sleep(1)
        print(f"❌ Timeout sur {url}")
        return False

    def stop(self):
        """Coupe tout."""
        print("🛑 Arrêt des serveurs...")
        if self.process_text: self.process_text.kill()
        if self.process_vision: self.process_vision.kill()