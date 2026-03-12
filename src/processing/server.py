import subprocess
import time
import requests
import yaml
import os
import sys
from pathlib import Path
import shlex

class LLMServerManager:
    def __init__(self, config_path="config/config.yaml"):
        self.process_text = None
        self.config = self._load_config(config_path)
        
        # Création du dossier logs s'il n'existe pas
        os.makedirs("logs", exist_ok=True)
        
    def _load_config(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_executable_path(self, key):
        platform_key = 'linux' if sys.platform == 'linux' else 'win'
        path_str = self.config['executables'][key][platform_key]
        return Path(path_str).absolute()

    def start(self):
        print("🚀 Initialisation du serveur IA...")
        ok_text = self._start_server_text()
        return ok_text

    def _check_binary_deps(self, exe_path):
        if sys.platform != 'linux': return
        # Simple check silencieux maintenant que tu as compilé
        pass

    def _start_server_text(self):
        exe_path = self._get_executable_path('llama_server')
        model_path = Path(self.config['models']['llm']['qwen3.5_0_8b']).resolve()
        
        print(f"   -> Démarrage Llama Server (8084)...")
        print(f"      Modèle : {model_path}")

        args_template = self.config['executables']['llama_server']['args']
        cmd_args = args_template.format(model_path=str(model_path))
        
        full_cmd = [str(exe_path)] + cmd_args.split()

        self.process_text = self._launch_process(full_cmd, "server_llama")
        return self._wait_for_url("http://localhost:8084/v1/models", self.process_text)


    def _launch_process(self, command, name):
        creation_flags = 0
        if os.name == 'nt':
            creation_flags = subprocess.CREATE_NO_WINDOW
        
        exe_path = Path(command[0])

        if sys.platform == 'linux':
            try: os.chmod(exe_path, 0o755)
            except OSError: pass

        # --- ENVIRONNEMENT ---
        my_env = os.environ.copy()
        if sys.platform == 'linux':
            exe_dir = str(exe_path.parent.absolute())
            current_ld = my_env.get("LD_LIBRARY_PATH", "")
            my_env["LD_LIBRARY_PATH"] = f"{exe_dir}:{current_ld}"

        # --- LOG FILES ---
        log_file_path = f"logs/{name}.log"
        
        try:
            log_out = open(log_file_path, "w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdout=log_out,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
                env=my_env,
                cwd=str(exe_path.parent) # On reste dans le dossier de l'exe (nécessaire pour ses libs)
            )
            return process
            
        except Exception as e:
            print(f"❌ Erreur critique lancement EXE ({command[0]}) : {e}")
            return None

    def _wait_for_url(self, url, process, timeout=60): # Timeout augmenté pour le chargement du gros modèle
        start_time = time.time()
        print(f"      ⏳ Attente disponibilité : {url}")
        
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                ret_code = process.returncode
                print(f"❌ FATAL : Le processus est mort immédiatement ! (Code: {ret_code})")
                print(f"👉 Regarde le fichier logs/server_text.log")
                return False

            try:
                requests.get(url, timeout=1)
                print(f"      ✅ Serveur prêt !")
                return True
            except requests.exceptions.RequestException:
                time.sleep(1)
                
        print(f"❌ Timeout sur {url} (Le modèle est trop long à charger ou manque de RAM)")
        return False

    def stop(self):
        print("🛑 Arrêt du serveur...")
        if self.process_text: 
            try:
                self.process_text.kill()
            except Exception:
                pass
                
        # S'assurer que le processus est bien tué sous Windows
        if os.name == 'nt':
            try:
                subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe', '/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass