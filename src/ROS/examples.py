#!/usr/bin/env python3
from PlayAudio import AudioController
from Display import DisplayController
from Moove import MoveController
import rospy

def main():
    # N'oublie pas de configurer IP/User/Pass dans Transfer.py !
    audio = AudioController()
    screen = DisplayController()
    move = MoveController()
    
    print("Démo transfert automatique...")

    # 1. Utilisation classique (Ressources internes)
    move.emotion("QT/happy")
    audio.say("Regarde mon écran, je vais afficher ton image.")
    rospy.sleep(3)

    # 2. Utilisation AVANCÉE (Fichiers locaux sur ton PC)
    # Le script va envoyer 'logo.png' au robot, l'appeler 'image_stream.png' et l'afficher
    print("Envoi de l'image...")
    screen.show_image("logo.png") 
    
    # Le script va envoyer 'musique.mp3' au robot, l'appeler 'audio_stream.mp3' et le jouer
    print("Envoi de la musique...")
    audio.play("musique.mp3")

    # On fait bouger le robot pendant la musique
    move.gesture("QT/dance")

    rospy.spin()

if __name__ == "__main__":
    main()