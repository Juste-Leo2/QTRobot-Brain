# QTRobot-Brain

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![English](https://img.shields.io/badge/Lang-English-red)](../README.md)

**Statut** : En développement actif. Le code est expérimental, instable et majoritairement non documenté.  
**Objectif** : Construire un pipeline complet à faibles ressources pour l'interaction homme-machine en temps réel sur le robot QT.

---

## Structure du projet
Le projet est divisé en **deux composants principaux** :

### 1. Module de Reconnaissance des Émotions  
Construction d'un système de détection d'émotions pour rendre le robot QT plus expressif et conscient du contexte.

**Dépôts Associés** :
- [AffectiveTRM](https://github.com/Juste-Leo2/AffectiveTRM) : Précédemment utilisé pour la reconnaissance multimodale des émotions.  
  > **Remarque** : Ce dépôt n'est pas utilisé dans le pipeline final car les résultats n'étaient pas concluants, mais il est conservé pour référence.

**Progression** :  
- [x] Collecte du jeu de données  
- [x] Entraînement du modèle  
- [x] Test du modèle  

---

### 2. Pipeline Complet d'Interaction à Faibles Ressources  
Un pipeline hors ligne et peu gourmand en calcul permettant une interaction en temps réel sur du matériel embarqué.

**Composants du pipeline** :
- [x] **Backend LLM ([llama.cpp](https://github.com/ggml-org/llama.cpp))**  
- [x] **Speech-to-text ([Vosk](https://github.com/alphacep/vosk-api))**
- [x] **Text-to-speech ([Piper](https://github.com/OHF-Voice/piper1-gpl))**
- [x] **Construction d'agent personnalisé avec prompts personnalisés**  
- [x] **Couche d'orchestration unifiée (Nouvelle Architecture "Executor")**  

---

## Fonctionnalités en Développement & Roadmap

- [x] Support Windows & Linux (x64) via GitHub Workflows
- [x] Script testé directement sur le robot QT
- [x] **Nouvelle Architecture** : Exécuteur threadé avec gestion des actions par file d'attente (queue) pour la sécurité des threads.
- [x] **Veste Connectée** : Support via le flag `--JKT` utilisant [QT-Touch](https://github.com/Juste-Leo2/QT-Touch).
- [x] **Support API** : Intégration optionnelle de l'API Google Gemini via `--API`.
- [x] **Mode Headless** : Exécution sans interface graphique via `--no-ui`.
- [x] Argument `--QT` dans `main.py` : à utiliser lors de l'exécution sur le robot.
- [x] Argument `--pytest` dans `main.py` : utilisé pour tester le pipeline (hors UI).
- [x] Argument `--ci-mode` dans `download_model.py` : remplace le gros modèle 8B par un modèle plus petit de 350M.
- [x] Argument `--name` : Définir un mot de réveil (wake-word) personnalisé.
- [ ] **Changement de langue** : Possibilité de basculer entre le français et l'anglais.
- [ ] **Sélection de la voix** : Option pour changer le genre de la voix (Homme/Femme).
- [ ] **Support API OpenAI** : Prise en charge de l'API d'OpenAI (actuellement le mode `--API` ne supporte que les modèles de Google).
- [ ] **Documentation Utilisateur** : Création d'un guide complet pour faciliter la prise en main et l'utilisation du dépôt par les nouveaux utilisateurs.

## Pourquoi rendre ce projet public ?
Développer ouvertement pour la **transparence, les retours et la responsabilité**.  
**Contributeurs bienvenus** — n'hésitez pas à ouvrir une issue si vous souhaitez aider !