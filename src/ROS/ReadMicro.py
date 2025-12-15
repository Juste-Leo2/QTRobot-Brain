#!/usr/bin/env python3
import rospy
import queue
import threading
from audio_common_msgs.msg import AudioData

class AudioStreamer:
    """
    Classe pour récupérer l'audio du robot QT en arrière-plan via ROS.
    """
    def __init__(self, topic_name='/qt_respeaker_app/channel0', queue_size=50):
        # On initialise le nœud ROS s'il ne l'est pas déjà
        if not rospy.core.is_initialized():
            rospy.init_node('audio_client_local', anonymous=True)
            
        self.topic_name = topic_name
        self.audio_queue = queue.Queue(maxsize=queue_size)
        self.subscriber = None
        self.is_listening = False

    def _audio_callback(self, msg):
        """Fonction interne appelée par ROS à chaque paquet audio"""
        if not self.is_listening:
            return

        try:
            # On récupère les données brutes (bytes)
            raw_data = bytes(msg.data)
            
            # Si la queue est pleine, on vide le plus vieux paquet pour garder le temps réel
            if self.audio_queue.full():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.audio_queue.put(raw_data)
        except Exception as e:
            rospy.logerr(f"Erreur audio callback: {e}")

    def start_listening(self):
        """Active l'écoute du topic ROS"""
        if self.subscriber is None:
            self.subscriber = rospy.Subscriber(self.topic_name, AudioData, self._audio_callback)
            self.is_listening = True
            rospy.loginfo(f"Flux audio activé sur {self.topic_name}")

    def stop_listening(self):
        """Coupe l'écoute pour économiser le CPU du robot"""
        if self.subscriber is not None:
            self.subscriber.unregister()
            self.subscriber = None
        
        self.is_listening = False
        # On vide le buffer pour ne pas traiter du vieux son au redémarrage
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        rospy.loginfo("Flux audio coupé.")

    def get_audio_chunk(self):
        """
        Récupère un morceau d'audio. 
        Retourne None si pas d'audio disponible immédiatement.
        """
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def is_active(self):
        return not rospy.is_shutdown()