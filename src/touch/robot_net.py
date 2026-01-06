import paramiko
import socket
import time
import threading
import os

class RaspberryManager:
    def __init__(self, ip, user, password, script_path, venv_path, port=65432):
        self.ip = ip
        self.user = user
        self.password = password
        self.script_path = script_path
        self.venv_path = venv_path
        self.port = port
        
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        self.sock = None
        self.last_data = None
        self.running = False

    def _listen_loop(self):
        """Boucle d'écoute."""
        while self.running:
            try:
                data = self.sock.recv(1024)
                if data:
                    self.last_data = data.decode('utf-8')
                else:
                    print("[INFO] Le Raspberry a coupé la connexion.")
                    self.running = False
                    break
            except socket.timeout:
                continue
            except Exception as e:
                if self.running: print(f"[ERREUR LOOP] {e}")
                break

    def connect_and_start(self):
        try:
            print(f"[INFO] Connexion SSH à {self.ip}...")
            self.ssh.connect(self.ip, username=self.user, password=self.password)
            
            # 1. Kill old process
            self.ssh.exec_command(f"pkill -f {self.script_path}")
            time.sleep(1)
            
            # 2. Construction de la commande
            # IMPORTANT : On récupère le dossier parent du script pour faire un 'cd'
            # Sinon il ne trouvera pas le fichier .onnx
            remote_dir = os.path.dirname(self.script_path) 
            
            # Commande : cd DOSSIER && source ACTIVATE && python SCRIPT
            cmd = f"cd {remote_dir} && . {self.venv_path} && python3 {self.script_path}"
            
            print(f"[INFO] Exécution: {cmd}")
            
            # On lance la commande (stdin, stdout, stderr pour debug si besoin)
            stdin, stdout, stderr = self.ssh.exec_command(cmd)
            
            # 3. Attente démarrage serveur
            print("[INFO] Attente du démarrage du serveur RPi...")
            time.sleep(4) 
            
            # 4. Connexion Socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.ip, self.port))
            self.sock.settimeout(None) # Mode bloquant pour le thread
            
            # 5. Thread
            self.running = True
            self.thread = threading.Thread(target=self._listen_loop)
            self.thread.daemon = True
            self.thread.start()
            
            print("[OK] Connecté et prêt.")
            return True

        except Exception as e:
            print(f"[CRITIQUE] Echec connexion : {e}")
            # Si ça plante, on essaie de lire l'erreur renvoyée par le SSH
            try:
                if 'stderr' in locals():
                    print(f"Log RPi : {stderr.read().decode()}")
            except: pass
            return False

    def get_data(self, clear_after=True):
        data = self.last_data
        if clear_after:
            self.last_data = None
        return data

    def stop(self):
        self.running = False
        try:
            if self.sock: self.sock.close()
            self.ssh.exec_command(f"pkill -f {self.script_path}")
            self.ssh.close()
        except: pass