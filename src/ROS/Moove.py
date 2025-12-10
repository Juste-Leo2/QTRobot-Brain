#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

class MoveController:
    """
    Contrôle les gestes et les émotions complexes du robot.
    """
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('move_controller_mod', anonymous=True)

        self.gesture_pub = rospy.Publisher('/qt_robot/gesture/play', String, queue_size=10)
        self.emotion_pub = rospy.Publisher('/qt_robot/emotion/show', String, queue_size=10)
        rospy.sleep(0.5)

    def gesture(self, gesture_name):
        """
        Joue un geste moteur uniquement.
        Ex: "QT/wave", "QT/bye"
        """
        rospy.loginfo(f"Gesture: {gesture_name}")
        self.gesture_pub.publish(gesture_name)

    def emotion(self, emotion_name):
        """
        Joue une émotion (Geste + Son + Visage combinés).
        Ex: "QT/happy", "QT/angry"
        """
        rospy.loginfo(f"Emotion: {emotion_name}")
        self.emotion_pub.publish(emotion_name)