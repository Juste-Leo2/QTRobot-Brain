# Fichier: ui.py

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2 # OpenCV pour la caméra

# Définition des thèmes pour l'application
ctk.set_appearance_mode("System")  # Modes: "System" (défaut), "Dark", "Light"
ctk.set_default_color_theme("blue") # Thèmes: "blue" (défaut), "green", "dark-blue"

class UI(ctk.CTk):
    """
    Une classe pour créer une interface de prototypage avec CustomTkinter.
    - À gauche : un visualiseur d'image (flux webcam ou image chargée).
    - À droite : trois zones de texte.
    - En bas : une grande zone de texte (pour logs, console, etc.).
    """
    def __init__(self):
        super().__init__()

        # --- Configuration de la fenêtre principale ---
        self.title("Interface de Prototypage")
        self.geometry("1280x720")

        # --- Configuration de la grille principale (layout) ---
        # 1ère colonne (image) prendra 2/3 de la largeur
        # 2ème colonne (zones de texte) prendra 1/3 de la largeur
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        # 1ère ligne (contenu principal) prendra la majorité de la hauteur
        # 2ème ligne (zone de texte du bas) prendra le reste
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0) # Poids plus faible

        # --- Partie gauche : Caméra / Image ---
        self.left_frame = ctk.CTkFrame(self, corner_radius=10)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.btn_load_image = ctk.CTkButton(self.left_frame, text="Charger une image", command=self.load_image)
        self.btn_load_image.grid(row=0, column=0, padx=10, pady=10)

        # Étiquette pour afficher l'image ou le flux vidéo
        self.image_label = ctk.CTkLabel(self.left_frame, text="Le flux de la caméra apparaîtra ici.\nSi aucune caméra n'est détectée,\nchargez une image.")
        self.image_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Initialisation de la capture vidéo
        self.cap = cv2.VideoCapture(0) # 0 est généralement la webcam par défaut
        self.update_camera_feed()

        # --- Partie droite : Trois zones de texte ---
        self.right_frame = ctk.CTkFrame(self, corner_radius=10)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.label_1 = ctk.CTkLabel(self.right_frame, text="Zone de texte 1", font=ctk.CTkFont(size=14, weight="bold"))
        self.label_1.pack(padx=10, pady=(10, 5), anchor="w")
        self.textbox_1 = ctk.CTkTextbox(self.right_frame, height=100)
        self.textbox_1.pack(padx=10, pady=5, fill="x", expand=True)

        self.label_2 = ctk.CTkLabel(self.right_frame, text="Zone de texte 2", font=ctk.CTkFont(size=14, weight="bold"))
        self.label_2.pack(padx=10, pady=(20, 5), anchor="w")
        self.textbox_2 = ctk.CTkTextbox(self.right_frame, height=100)
        self.textbox_2.pack(padx=10, pady=5, fill="x", expand=True)

        self.label_3 = ctk.CTkLabel(self.right_frame, text="Zone de texte 3", font=ctk.CTkFont(size=14, weight="bold"))
        self.label_3.pack(padx=10, pady=(20, 5), anchor="w")
        self.textbox_3 = ctk.CTkTextbox(self.right_frame, height=100)
        self.textbox_3.pack(padx=10, pady=5, fill="x", expand=True)
        
        # --- Partie basse : Zone de texte longue ---
        self.bottom_frame = ctk.CTkFrame(self, corner_radius=10)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.bottom_label = ctk.CTkLabel(self.bottom_frame, text="Console / Logs", font=ctk.CTkFont(size=14, weight="bold"))
        self.bottom_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.bottom_textbox = ctk.CTkTextbox(self.bottom_frame, height=120)
        self.bottom_textbox.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

        # Gérer la fermeture de la fenêtre pour libérer la caméra
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_camera_feed(self):
        """Met à jour l'étiquette avec une nouvelle image de la webcam."""
        ret, frame = self.cap.read()
        if ret:
            # Convertir l'image de OpenCV (BGR) vers PIL (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Créer une CTkImage et la mettre à jour
            # On ajuste la taille pour qu'elle s'adapte au label
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
            self.image_label.configure(image=ctk_image, text="")
            self.image_label.image = ctk_image # Garder une référence
        
        # Répéter cette fonction après 15ms
        self.after(15, self.update_camera_feed)

    def load_image(self):
        """Ouvre une boîte de dialogue pour choisir et afficher une image statique."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Tous les fichiers", "*.*")]
        )
        if file_path:
            if self.cap.isOpened():
                self.cap.release()
                self.btn_load_image.configure(text="Réactiver la caméra", command=self.restart_camera)

            pil_image = Image.open(file_path)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
            self.image_label.configure(image=ctk_image, text="")
            self.image_label.image = ctk_image

    def restart_camera(self):
        """Réinitialise la capture vidéo et relance le flux."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.image_label.configure(image=None, text="Impossible de réactiver la caméra.")
            return
            
        self.btn_load_image.configure(text="Charger une image", command=self.load_image)
        self.update_camera_feed()

    def on_closing(self):
        """Libère les ressources (caméra) avant de fermer l'application."""
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = UI()
    app.mainloop()