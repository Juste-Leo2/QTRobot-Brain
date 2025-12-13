#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState

class HeadController:
    """
    Contrôle la tête du robot pour suivre un visage (Face Tracking).
    """
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('head_controller_mod', anonymous=True)

        self.joint_pub = rospy.Publisher('/qt_robot/joints/state/target', JointState, queue_size=10)
        
        # Limites (Radians)
        self.yaw_limit = (-1.2, 1.2)   # Gauche/Droite
        self.pitch_limit = (-0.4, 0.4) # Haut/Bas
        
        self.current_yaw = 0.0
        self.current_pitch = 0.0

        rospy.sleep(0.5)

    def move_head_to_center(self, face_x, face_y, img_w, img_h):
        """
        Asservissement pour centrer le visage.
        """
        center_x = img_w / 2
        center_y = img_h / 2

        # Gain P (Vitesse)
        kp_yaw = 0.001
        kp_pitch = 0.001

        error_x = center_x - face_x
        error_y = center_y - face_y

        # Deadzone (évite les micro-mouvements)
        if abs(error_x) < 40: error_x = 0
        if abs(error_y) < 40: error_y = 0

        if error_x == 0 and error_y == 0:
            return

        # Mise à jour de la consigne
        self.current_yaw += (error_x * kp_yaw)
        self.current_pitch -= (error_y * kp_pitch) # Pitch souvent inversé

        # Saturation
        self.current_yaw = max(min(self.current_yaw, self.yaw_limit[1]), self.yaw_limit[0])
        self.current_pitch = max(min(self.current_pitch, self.pitch_limit[1]), self.pitch_limit[0])

        self.send_command(self.current_yaw, self.current_pitch)

    def send_command(self, yaw, pitch):
        msg = JointState()
        msg.name = ["HeadYaw", "HeadPitch"]
        msg.position = [float(yaw), float(pitch)]
        self.joint_pub.publish(msg)