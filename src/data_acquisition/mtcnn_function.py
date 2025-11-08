# src/data_acquisition/mtcnn_function.py

import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

# Initialiser le détecteur une seule fois lors de l'importation du module
detector = MTCNN()

def detect_faces(image: np.ndarray) -> list:
    """
    Détecte les visages dans une image donnée.
    :param image: Une image (frame) au format NumPy array (BGR).
    :return: Une liste de dictionnaires, chaque dictionnaire contenant les infos d'un visage.
    """
    # La détection se fait sur une image convertie en RGB
    faces = detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return faces

def draw_faces(image: np.ndarray, faces: list):
    """
    Dessine des rectangles autour des visages détectés sur une image.
    :param image: L'image sur laquelle dessiner.
    :param faces: La liste des visages retournée par detect_faces.
    """
    for face in faces:
        if face['confidence'] > 0.95:
            x, y, width, height = face['box']
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Fonction principale pour la démonstration en temps réel
def run_realtime_detection():
    """Lance la capture vidéo et la détection en temps réel."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de recevoir l'image.")
            break

        detected_faces = detect_faces(frame)
        draw_faces(frame, detected_faces)

        cv2.imshow('Detection de visage - Appuyez sur Q pour quitter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_realtime_detection()