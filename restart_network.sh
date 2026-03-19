#!/bin/bash

echo "=================================================="
echo "🔄 Redémarrage du service WLAN0 sur le robot QT..."
echo "=================================================="

# 1. Vérification de sshpass (nécessaire pour envoyer le mot de passe)
if ! command -v sshpass &> /dev/null
then
    echo "❌ ERREUR: 'sshpass' n'est pas installé sur cet ordinateur."
    echo "👉 Installez-le en tapant : sudo apt-get install sshpass"
    exit 1
fi

# 2. Connexion SSH et exécution du redémarrage
# - sshpass tape le mot de passe SSH
# - echo 'qtrobot' | sudo -S tape le mot de passe pour les droits administrateur
sshpass -p 'qtrobot' ssh -o StrictHostKeyChecking=no developer@192.168.100.1 "echo 'qtrobot' | sudo -S systemctl restart qt_wlan0_client.service"

# 3. Pause de sécurité
echo "⏳ Commande envoyée. Attente de 3 secondes pour que le réseau se stabilise..."
sleep 3

echo "✅ Terminé ! Le réseau devrait être de nouveau opérationnel."