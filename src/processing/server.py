import subprocess
import time
import requests
import yaml
import os
import sys
from pathlib import Path

class LLMServerManager:
    def __init__(self, config_path="config/config.yaml"):
        self.process_text = None
        self.process_vision = None
        self.config = self._load_config(config_path)
        
    def _load_config(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_executable_path(self, key):
        platform_key = 'linux' if sys.platform == 'linux' else 'win'
        return Path(self.config['executables'][key][platform_key])

    def start(self):
        print("🚀 Initialisation des serveurs IA...")
        ok_text = self._start_server_text()
        ok_vision = self._start_server_vision()
        return ok_text and ok_vision

    def _start_server_text(self):
        exe_path = self._get_executable_path('llama_server')
        model_path = Path(self.config['models']['llm']['lfm_8b'])
        
        print(f"   -> Démarrage Llama Texte (8084) [{exe_path}]...")
        
        args_template = self.config['executables']['llama_server']['args']
        cmd_args = args_template.format(model_path=str(model_path))
        full_cmd = [str(exe_path)] + cmd_args.split()

        self.process_text = self._launch_process(full_cmd)
        return self._wait_for_url(self.config['llm_server']['url'])

    def _start_server_vision(self):
        exe_path = self._get_executable_path('llama_server_vision')
        model_path = Path(self.config['models']['llm']['LFM2-VL-450M-Q4'])
        mmproj_path = Path(self.config['models']['llm']['mmproj-LFM2-VL-450M-Q8'])
        
        print(f"   -> Démarrage Llama Vision (8088) [{exe_path}]...")

        args_template = self.config['executables']['llama_server_vision']['args']
        cmd_args = args_template.format(
            model_path=str(model_path),
            model_path_mmproj=str(mmproj_path)
        )
        full_cmd = [str(exe_path)] + cmd_args.split()

        self.process_vision = self._launch_process(full_cmd)
        base_url = "http://localhost:8088/health" 
        return self._wait_for_url(base_url)

    def _launch_process(self, command):
        creation_flags = 0
        if os.name == 'nt':
            creation_flags = subprocess.CREATE_NO_WINDOW
        
        # S'assurer que le fichier est exécutable sous Linux
        if sys.platform == 'linux':
            try:
                os.chmod(command[0], 0o755)
            except OSError:
                pass

        # --- FIX LINUX LIBS ---
        # On ajoute le dossier contenant l'exe au LD_LIBRARY_PATH
        # pour qu'il trouve libmtmd.so et autres
        my_env = os.environ.copy()
        if sys.platform == 'linux':
            exe_dir = str(Path(command[0]).parent.absolute())
            current_ld = my_env.get("LD_LIBRARY_PATH", "")
            my_env["LD_LIBRARY_PATH"] = f"{exe_dir}:{current_ld}"

        try:
            return subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
                env=my_env # On passe l'environnement modifié
            )
        except Exception as e:
            print(f"❌ Erreur lancement EXE ({command[0]}) : {e}")
            return None

    def _wait_for_url(self, url, timeout=30):
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
        print("🛑 Arrêt des serveurs...")
        if self.process_text: self.process_text.kill()
        if self.process_vision: self.process_vision.kill()