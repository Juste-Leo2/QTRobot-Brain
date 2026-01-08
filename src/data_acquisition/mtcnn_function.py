#src/data_acquisition/mtcnn_function.py
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import math

# Initialiser le détecteur une seule fois
# On le met en variable globale pour ne pas le recharger à chaque appel
detector = MTCNN()

def detect_faces(image: np.ndarray) -> list:
    """
    Détecte les visages avec protection contre les crashs (bords d'image).
    """
    if image is None or image.size == 0:
        return []

    try:
        # La conversion RGB est nécessaire pour MTCNN
        img_rgb = image
        faces = detector.detect_faces(img_rgb)
        return faces
    except ValueError:
        # Capture l'erreur "Output shape: (0, 48, 48, 3)" quand un visage est coupé
        return []
    except Exception as e:
        print(f"⚠️ Erreur MTCNN ignorée: {e}")
        return []

def get_face_center(box):
    """Calcule le centre (x, y) d'une bounding box [x, y, w, h]"""
    x, y, w, h = box
    return (x + w / 2, y + h / 2)

def calculate_distance(p1, p2):
    """Distance Euclidienne entre deux points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def select_priority_face(faces, last_center=None):
    """
    Sélectionne le visage prioritaire.
    - Si last_center existe : Choisit le visage dont le centre est le plus proche.
    - Si last_center est None : Choisit le visage le plus grand (le plus proche de la caméra).
    
    Retourne: (best_face, new_center) ou (None, None)
    """
    if not faces:
        return None, None

    best_face = None
    min_dist = float('inf')
    max_area = -1

    # Si on a une historique de position, on cherche le visage le plus proche spatialement
    if last_center is not None:
        for face in faces:
            center = get_face_center(face['box'])
            dist = calculate_distance(last_center, center)
            
            # Seuil de "saut" (optionnel) : si le visage a bougé de trop loin d'un coup, on pourrait l'ignorer
            # Ici on prend juste le minimum
            if dist < min_dist:
                min_dist = dist
                best_face = face
        
        # Si on a trouvé un visage proche, on retourne son nouveau centre
        if best_face:
            return best_face, get_face_center(best_face['box'])

    # Sinon (premier lancement ou perte de tracking), on prend le plus gros visage (confiance/taille)
    # MTCNN retourne souvent [x, y, w, h], l'aire est w * h
    for face in faces:
        _, _, w, h = face['box']
        area = w * h
        if area > max_area:
            max_area = area
            best_face = face
            
    if best_face:
        return best_face, get_face_center(best_face['box'])
    
    return None, None

def draw_faces(image: np.ndarray, faces: list, priority_index=None):
    """
    Dessine les visages. Si priority_index est fourni, dessine celui-ci en Vert, les autres en Rouge.
    """
    for i, face in enumerate(faces):
        if face['confidence'] > 0.90:
            x, y, width, height = face['box']
            
            # Couleur : Vert si c'est le visage tracké, Rouge sinon
            color = (0, 0, 255) # Rouge par défaut
            if priority_index is not None and i == priority_index:
                color = (0, 255, 0) # Vert
            elif priority_index is None: # Si pas de distinction, tout en vert
                color = (0, 255, 0)

            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)

# --- BLOC DE TEST ---
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    last_center = None # Variable d'état pour le tracking

    while True:
        ret, frame = cap.read()
        if not ret: break

        detected_faces = detect_faces(frame)
        
        # Logique de Tracking
        priority_face, new_center = select_priority_face(detected_faces, last_center)
        
        if priority_face:
            last_center = new_center # Mise à jour de la position
            # On dessine (astuce: on doit retrouver l'index pour draw_faces ou juste dessiner ici)
            x, y, w, h = priority_face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "TARGET", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            last_center = None # Perdu

        cv2.imshow('Tracking Prioritaire', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()