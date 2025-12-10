#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from Transfer import FileTransfer # On importe notre outil

class AudioController:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('audio_controller_mod', anonymous=True)

        self.tts_pub = rospy.Publisher('/qt_robot/speech/say', String, queue_size=10)
        self.audio_pub = rospy.Publisher('/qt_robot/audio/play', String, queue_size=10)
        
        # Initialisation du gestionnaire de transfert
        self.transfer = FileTransfer()
        rospy.sleep(0.5)

    def say(self, text):
        rospy.loginfo(f"TTS: {text}")
        self.tts_pub.publish(text)

    def play(self, filename):
        """
        - Si filename commence par "QT/", c'est un son interne.
        - Sinon, c'est un fichier sur TON PC qu'on envoie au robot.
        """
        final_path = filename

        if not filename.startswith("QT/"):
            # C'est un fichier local, on l'envoie d'abord
            remote_path = self.transfer.send(filename, file_prefix="audio_stream")
            if remote_path:
                final_path = remote_path
            else:
                return # Echec transfert

        rospy.loginfo(f"Audio Play: {final_path}")
        self.audio_pub.publish(final_path)