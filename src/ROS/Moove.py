#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

class MoveController:
    """
    Contrôle les gestes (Moteurs) et les émotions (Ecran/Visage) du robot QT via ROS.
    Utilise des Publishers (Topics) pour une exécution non-bloquante (Concurrent).
    """
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('move_controller_mod', anonymous=True)

        # On respecte les topics officiels de LuxAI vus dans ton exemple
        self.gesture_pub = rospy.Publisher('/qt_robot/gesture/play', String, queue_size=10)
        self.emotion_pub = rospy.Publisher('/qt_robot/emotion/show', String, queue_size=10)
        
        # Petit temps de pause pour laisser ROS établir les connexions
        rospy.sleep(0.5)

    def gesture(self, gesture_name):
        """
        Joue un geste moteur (Ex: QT/wave, QT/hi).
        C'est non-bloquant : le code continue immédiatement.
        """
        # Sécurité : si le nom est vide
        if not gesture_name: return

        rospy.loginfo(f" ROS GESTURE: {gesture_name}")
        self.gesture_pub.publish(str(gesture_name))

    def emotion(self, emotion_name):
        """
        Change le visage sur l'écran (Ex: QT/happy, QT/sad).
        """
        if not emotion_name: return

        rospy.loginfo(f" ROS EMOTION: {emotion_name}")
        self.emotion_pub.publish(str(emotion_name))