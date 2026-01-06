import socket
import time
import collections
from collections import Counter
import numpy as np
import onnxruntime as ort
from gpiozero import MCP3008
import sys
import os
import threading

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 65432

# Paramètres IA / Signal
ACQUISITION_FREQ = 1000  
DOWNSAMPLING = 10        
MODEL_PATH = "veste_model.onnx"
CLASSES = ["Rien", "Tape", "Pincement", "Frottement"]
WINDOW_SIZE = 100        
INFERENCE_STRIDE = 5     
VOTE_WINDOW_SIZE = 5     
SCORE_THRESHOLD = 0.80   
SEUIL_STABILITE = 0.008  

# Paramètres de confirmation et délai
CONFIRMATION_SIZE = 3       # 3 votes majoritaires identiques pour valider
COOLDOWN_DELAY = 10.0       # 10 secondes de pause après un envoi

class GestureThread(threading.Thread):
    def __init__(self, client_socket):
        super().__init__()
        self.client_socket = client_socket
        self.running = True
        self.adc = MCP3008(channel=0, clock_pin=12, mosi_pin=20, miso_pin=16, select_pin=21)
        
        # --- 1. CHARGEMENT MODELE (Une seule fois ici) ---
        print(f"[Thread] Chargement modele: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"ERREUR: {MODEL_PATH} introuvable")
            self.running = False
            return

        try:
            self.session = ort.InferenceSession(MODEL_PATH)
            self.input_name = self.session.get_inputs()[0].name
            print("[Thread] Modele ONNX charge avec succes.")
        except Exception as e:
            print(f"[Thread] Erreur ONNX: {e}")
            self.running = False

        # Buffers
        self.raw_buffer = collections.deque(maxlen=WINDOW_SIZE)
        self.vote_buffer = collections.deque(maxlen=VOTE_WINDOW_SIZE)
        self.confirmation_buffer = collections.deque(maxlen=CONFIRMATION_SIZE)
        self.accumulator = []

        # Initialisation "Intelligente" des buffers pour éviter le faux positif au start
        self._reset_buffers(initial_fill=True)

    def _reset_buffers(self, initial_fill=False):
        """
        Vide les buffers. 
        Si initial_fill=True, on lit le capteur pour remplir le buffer avec la valeur ACTUELLE.
        Cela évite le saut 0 -> 500 qui déclenche un 'Tape' au démarrage.
        """
        self.vote_buffer.clear()
        for _ in range(VOTE_WINDOW_SIZE): self.vote_buffer.append(0)
        
        self.confirmation_buffer.clear()
        self.accumulator = []
        
        if initial_fill:
            # On lit une valeur stable pour remplir le buffer
            # On fait une petite moyenne sur 10 lectures pour être propre
            vals = [self.adc.value for _ in range(10)]
            avg_start = sum(vals) / len(vals)
            
            self.raw_buffer.clear()
            # On remplit tout le buffer avec cette valeur moyenne
            # Comme ça, la dérivée est de 0 (plat) -> Pas de détection
            for _ in range(WINDOW_SIZE): self.raw_buffer.append(avg_start)

    def run(self):
        print("[Thread] Demarrage de la boucle d'inference...")
        points_processed_counter = 0 
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Acquisition
                val = self.adc.value
                self.accumulator.append(val)
                
                # Traitement (Downsampling vers 100Hz)
                if len(self.accumulator) >= DOWNSAMPLING:
                    avg_val = sum(self.accumulator) / len(self.accumulator)
                    self.accumulator = []
                    
                    self.raw_buffer.append(avg_val)
                    points_processed_counter += 1
                    
                    # Inférence (20Hz)
                    if points_processed_counter % INFERENCE_STRIDE == 0:
                        raw_signal = np.array(self.raw_buffer, dtype=np.float32)
                        ecart_type = np.std(raw_signal)
                        
                        instant_pred = 0
                        
                        # Optimisation : Si le signal est plat, on ne lance pas l'IA
                        if ecart_type >= SEUIL_STABILITE:
                            derivative = np.diff(raw_signal)
                            derivative = np.insert(derivative, 0, 0) # Pad pour garder la taille
                            input_data = derivative.reshape(1, 1, 100)
                            
                            outputs = self.session.run(None, {self.input_name: input_data})
                            probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
                            
                            if np.max(probs) > SCORE_THRESHOLD:
                                instant_pred = np.argmax(probs)
                        
                        # --- LOGIQUE DE VOTE ---
                        # 1. Vote court terme
                        self.vote_buffer.append(instant_pred)
                        winner_class, _ = Counter(self.vote_buffer).most_common(1)[0]
                        
                        # 2. Vote de confirmation (Sécurité)
                        self.confirmation_buffer.append(winner_class)

                        if len(self.confirmation_buffer) == CONFIRMATION_SIZE:
                            unique_votes = set(self.confirmation_buffer)
                            
                            # Si les 3 derniers votes majoritaires sont IDENTIQUES et NON NULS
                            if len(unique_votes) == 1:
                                confirmed_gesture = list(unique_votes)[0]
                                
                                if confirmed_gesture != 0:
                                    nom = CLASSES[confirmed_gesture]
                                    print(f">>> GESTE VALIDE : {nom}")
                                    
                                    # Envoi au PC
                                    try:
                                        self.client_socket.sendall(nom.encode('utf-8'))
                                    except (BrokenPipeError, ConnectionResetError):
                                        print("[Thread] Connexion perdue lors de l'envoi.")
                                        self.running = False
                                        break
                                    
                                    # --- PAUSE DE 10 SECONDES ---
                                    print(f"[Thread] Pause de {COOLDOWN_DELAY}s (Stabilisation)...")
                                    time.sleep(COOLDOWN_DELAY)
                                    
                                    # --- RESET TOTAL (Suppression historique) ---
                                    print("[Thread] Reprise. Reset des buffers.")
                                    self._reset_buffers(initial_fill=True)
                                    # On sort de la boucle de traitement pour ce tour
                                    continue 

                # Cadencement 1kHz
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / ACQUISITION_FREQ) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            except Exception as e:
                print(f"[Thread] Erreur fatale boucle: {e}")
                self.running = False

    def stop(self):
        self.running = False


# --- MAIN (GESTION RESEAU) ---
if __name__ == "__main__":
    print(f"--- Serveur Veste Demarre (PID: {os.getpid()}) ---")
    sys.stdout.flush()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print("En attente de connexion du PC...")
        sys.stdout.flush()
        
        # Bloquant jusqu'à connexion
        client, addr = server_socket.accept()
        print(f"PC Connecte : {addr}")
        
        # Lancement du Thread de détection
        # On lui passe le socket client pour qu'il puisse envoyer directement
        gesture_thread = GestureThread(client)
        gesture_thread.start()
        
        # Le main attend juste que le thread finisse (ou crash)
        while gesture_thread.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nArret manuel.")
    except Exception as e:
        print(f"Erreur Main: {e}")
    finally:
        print("Fermeture connexions...")
        if 'gesture_thread' in locals():
            gesture_thread.stop()
            gesture_thread.join()
        if 'client' in locals():
            client.close()
        server_socket.close()