import customtkinter as ctk
from PIL import Image

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class UI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Assistant IA - Vision & Chat")
        self.geometry("1280x720")

        # --- Layout ---
        self.grid_columnconfigure(0, weight=3) # La colonne image est plus large
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Cadre Image (Gauche) ---
        self.left_frame = ctk.CTkFrame(self, corner_radius=0)
        self.left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        
        # Label Image : On le centre
        self.image_label = ctk.CTkLabel(self.left_frame, text="Démarrage caméra...")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center") # Centré parfaitement

        # --- Cadre Chat (Droite) ---
        self.right_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.textbox_1 = self.create_textbox(self.right_frame, "Vous")
        self.textbox_2 = self.create_textbox(self.right_frame, "Outil")
        self.textbox_3 = self.create_textbox(self.right_frame, "IA")

        # --- Logs (Bas Droite) ---
        self.bottom_textbox = ctk.CTkTextbox(self.right_frame, height=150)
        self.bottom_textbox.pack(fill="x", pady=(20, 0))

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_textbox(self, parent, title):
        lbl = ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(weight="bold"))
        lbl.pack(anchor="w", pady=(5,0))
        box = ctk.CTkTextbox(parent, height=80)
        box.pack(fill="x", pady=(0,10))
        return box

    def mettre_a_jour_image(self, pil_image):
        """
        Reçoit une image PIL déjà redimensionnée et l'affiche.
        """
        # On crée l'objet CTkImage (obligatoire sur le main thread)
        # size=(w, h) est important, on reprend la taille de l'image envoyée
        w, h = pil_image.size
        ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(w, h))
        
        self.image_label.configure(image=ctk_img, text="")

    def on_closing(self):
        self.destroy()