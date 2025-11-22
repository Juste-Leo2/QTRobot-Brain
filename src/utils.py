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

def analyser_image_via_api(image, url_api):
    """
    Prend une image brute (OpenCV), la prépare (resize/base64) 
    et l'envoie à l'API Vision (Liquid AI ou autre).
    Retourne la description textuelle ou None en cas d'erreur.
    """
    if image is None:
        return None

    try:
        # 1. Redimensionnement Optimisé pour l'inférence (400x200)
        img_inference = cv2.resize(image, (400, 200))
        
        # 2. Encodage Base64
        _, buffer = cv2.imencode('.jpg', img_inference)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 3. Préparation Payload
        system_prompt = "You are a helpful multimodal assistant by Liquid AI."
        user_prompt = "ONE SHORT SENTENCE GENERAL description of the image above."

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ],
            "temperature": 0.1,       
            "min_p": 0.15,            
            "repeat_penalty": 1.05,   
            "max_tokens": 256,
            "stream": False
        }

        # 4. Envoi requête
        response = requests.post(url_api, json=payload, timeout=45)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            return content.replace("<|im_end|>", "").strip()
        else:
            print(f"[UTILS] Erreur API Vision: {response.status_code}")
            return None

    except Exception as e:
        print(f"[UTILS] Exception Vision: {e}")
        return None