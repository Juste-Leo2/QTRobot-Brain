#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from Transfer import FileTransfer

class DisplayController:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('display_controller_mod', anonymous=True)

        self.img_pub = rospy.Publisher('/qt_robot/screen/show_image', String, queue_size=10)
        self.txt_pub = rospy.Publisher('/qt_robot/screen/show_text', String, queue_size=10)
        
        self.transfer = FileTransfer()
        rospy.sleep(0.5)

    def show_image(self, filename):
        """
        - "QT/happy" -> image interne
        - "mon_image.jpg" -> envoie l'image locale au robot puis l'affiche
        """
        final_path = filename

        if not filename.startswith("QT/"):
            # Envoi de l'image locale vers le robot (écrase image_stream.jpg/png)
            remote_path = self.transfer.send(filename, file_prefix="image_stream")
            if remote_path:
                final_path = remote_path
            else:
                return

        rospy.loginfo(f"Display Image: {final_path}")
        self.img_pub.publish(final_path)

    def show_text(self, text):
        rospy.loginfo(f"Display Text: {text}")
        self.txt_pub.publish(text)