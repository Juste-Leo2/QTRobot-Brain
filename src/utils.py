# src/utils.py
import os
import subprocess
import datetime
import cv2
import base64
import requests

def obtenir_heure_formatee():
    """Retourne une phrase textuelle avec l'heure actuelle."""
    now = datetime.datetime.now()
    return now.strftime("Il est %H heures %M.")

def jouer_fichier_audio(chemin_fichier):
    """
    Joue un fichier audio (.wav) de manière compatible Windows/Linux.
    """
    if os.name == 'nt':
        # Windows via PowerShell
        cmd = f'powershell -c (New-Object Media.SoundPlayer "{chemin_fichier}").PlaySync()'
        subprocess.run(cmd, shell=True)
    else:
        # Linux via aplay
        os.system(f"aplay {chemin_fichier}")

def redimensionner_image_pour_ui(frame, target_width=800):
    """
    Convertit l'image BGR (OpenCV) en RGB et la redimensionne pour l'UI.
    Retourne un tableau numpy prêt pour PIL.
    """
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        ratio = target_width / w
        target_height = int(h * ratio)
        return cv2.resize(frame_rgb, (target_width, target_height))
    except Exception:
        return None

def encodage_image_base64_pour_api(image):
    """
    Prend une image brute (OpenCV), la redimensionne légèrement 
    pour optimiser les tailles Base64, et l'encode pour la passer
    aux modèles multimodaux via API.
    """
    if image is None:
        return None

    try:
        # Redimensionnement Optimisé pour l'inférence
        img_inference = cv2.resize(image, (640, 480))
        _, buffer = cv2.imencode('.jpg', img_inference)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

    except Exception as e:
        print(f"[UTILS] Exception Encodage Vision: {e}")
        return None