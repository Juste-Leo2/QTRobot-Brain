#!/usr/bin/env python3
import paramiko
import os
import rospy
import socket

class FileTransfer:
    def __init__(self):
        # --- CONFIGURATION RÉSEAU ---
        self.hostname = "192.168.100.1"
        
        # ---changement ici : developer au lieu de qtrobot---
        self.username = "developer"  
        self.password = "qtrobot"    # <-- Mets ici le mot de passe qui a marché dans le terminal
        
        # NOTE : On écrit quand même dans le dossier de qtrobot pour que ROS puisse le lire
        # Si ça plante "Permission Denied", on changera vers /tmp/
        self.remote_folder = "/tmp/" 
        
        self.ssh_available = False
        self._ensure_remote_folder()

    def _ensure_remote_folder(self):
        """Crée le dossier sur le Raspberry Pi via SSH"""
        client = None
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            client.connect(
                self.hostname, 
                username=self.username, 
                password=self.password, 
                timeout=3.0,
                look_for_keys=False, # Important : Ignore tes clés SSH perso
                allow_agent=False
            )
            
            # On utilise sudo -u qtrobot au cas où developer n'a pas les droits directs
            # Mais souvent developer a tous les droits.
            client.exec_command(f"mkdir -p {self.remote_folder}")
            self.ssh_available = True
            print(f"[Transfer] ✅ SSH connecté en tant que '{self.username}'.")
            
        except Exception as e:
            print(f"[Transfer] ⚠️ Echec SSH ({self.username}): {e}")
            self.ssh_available = False
        finally:
            if client: client.close()

    def send(self, local_path, file_prefix="temp"):
        if not self.ssh_available: return None
        if not os.path.exists(local_path): return None

        extension = os.path.splitext(local_path)[1]
        remote_filename = f"{file_prefix}{extension}"
        remote_path = os.path.join(self.remote_folder, remote_filename)

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            client.connect(
                self.hostname, 
                username=self.username, 
                password=self.password,
                timeout=5.0,
                look_for_keys=False,
                allow_agent=False
            )
            
            sftp = client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            client.close()
            
            # Important : On retourne le chemin complet pour que ROS le trouve
            return remote_path

        except Exception as e:
            print(f"[Transfer] ❌ Erreur envoi : {e}")
            return None