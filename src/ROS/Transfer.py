#!/usr/bin/env python3
import paramiko
import os
import rospy

class FileTransfer:
    def __init__(self):
        # --- CONFIGURATION INTERNE DU ROBOT ---
        # On cible le Raspberry Pi (QTRP) depuis le PC Principal (QTPC)
        self.hostname = "192.168.100.1"  # C'est l'IP du Raspberry Pi (voir ton schéma)
        
        # Les identifiants sont souvent les mêmes, mais vérifie si le RPI a un mdp différent
        self.username = "qtrobot"        
        self.password = "qtrobot"        # Parfois "qt" / "qt" sur les vieilles versions
        
        # Dossier de destination sur le Raspberry Pi
        self.remote_folder = "/home/qtrobot/cache_stream/" 

        self._ensure_remote_folder()

    def _ensure_remote_folder(self):
        """Crée le dossier sur le Raspberry Pi via SSH"""
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # On se connecte du QTPC vers le QTRP
            client.connect(self.hostname, username=self.username, password=self.password, timeout=2)
            client.exec_command(f"mkdir -p {self.remote_folder}")
            client.close()
        except Exception as e:
            rospy.logwarn(f"Erreur connexion SSH vers QTRP (100.1) : {e}")

    def send(self, local_path, file_prefix="temp"):
        """
        Prend un fichier sur le QTPC (local) et l'envoie sur le QTRP (remote).
        """
        if not os.path.exists(local_path):
            rospy.logerr(f"Fichier introuvable sur le QTPC : {local_path}")
            return None

        extension = os.path.splitext(local_path)[1]
        remote_filename = f"{file_prefix}{extension}"
        # Chemin final tel qu'il sera vu par le Raspberry Pi
        remote_path = os.path.join(self.remote_folder, remote_filename)

        try:
            transport = paramiko.Transport((self.hostname, 22))
            transport.connect(username=self.username, password=self.password)
            sftp = paramiko.SFTPClient.from_transport(transport)

            # Upload : QTPC -> QTRP
            # rospy.loginfo(f"Transfert interne QTPC -> QTRP : {local_path}")
            sftp.put(local_path, remote_path)
            
            sftp.close()
            transport.close()
            
            # On retourne le chemin distant, car c'est celui-là que ROS doit lire
            return remote_path

        except Exception as e:
            rospy.logerr(f"Echec transfert vers QTRP : {e}")
            return None