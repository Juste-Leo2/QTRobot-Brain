#src/data_acquisition/emotions.py
import torch
import cv2
import numpy as np
from collections import deque, Counter
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

class EmotionAnalyzer:
    def __init__(self, device=None, window_size=10):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        
        # Historique des émotions pour le lissage (Vote majoritaire)
        self.emotion_history = deque(maxlen=self.window_size)
        
        # Initialisation du modèle EmotiEffLib
        try:
            model_name = get_model_list()[0]
            self.fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=self.device)
            print(f"🧠 [EMOTION] Modèle chargé sur {self.device} (Lissage: {window_size} frames)")
        except Exception as e:
            print(f"❌ [EMOTION] Erreur chargement modèle: {e}")
            self.fer = None

    def process_emotion(self, frame_rgb, faces_det, scale_factor=1.0):
        """
        Analyse les émotions sur les visages détectés par MTCNN.
        :param frame_rgb: Image complète en RGB
        :param faces_det: Liste des dictionnaires retournée par detect_faces (MTCNN)
        :param scale_factor: Facteur d'échelle si la détection a été faite sur une image réduite
        :return: (smoothed_emotion, emotion_raw, box_tuple) pour le visage principal
        """
        if not self.fer or not faces_det:
            return None, None, None

        # On prend le visage le plus confiant (ou le plus grand)
        # MTCNN retourne 'box': [x, y, w, h]
        primary_face = max(faces_det, key=lambda x: x['confidence'])
        x, y, w, h = primary_face['box']

        # Ajustement de l'échelle si la détection a été faite sur une image réduite
        x = int(x * scale_factor)
        y = int(y * scale_factor)
        w = int(w * scale_factor)
        h = int(h * scale_factor)

        # Conversion en coordonnées x1, y1, x2, y2 pour le crop
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_rgb.shape[1], x + w), min(frame_rgb.shape[0], y + h)

        face_img = frame_rgb[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None, None, None

        try:
            # Prédiction sur le visage
            # predict_emotions attend une liste d'images
            emotions, _ = self.fer.predict_emotions([face_img], logits=False)
            current_emotion = emotions[0]

            # Ajout à l'historique
            self.emotion_history.append(current_emotion)

            # Vote majoritaire
            most_common = Counter(self.emotion_history).most_common(1)
            smoothed_emotion = most_common[0][0]

            return smoothed_emotion, current_emotion, (x1, y1, x2, y2)

        except Exception as e:
            print(f"⚠️ Erreur inférence émotion: {e}")
            return None, None, (x1, y1, x2, y2)