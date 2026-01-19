# QTRobot-Brain

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Français](https://img.shields.io/badge/Lang-Français-red)](docs/README_FR.md)

**Status**: In active development. Code is experimental, unstable, and mostly undocumented.  
**Goal**: Build a complete, low-resource pipeline for real-time human–machine interaction on the QT robot.

---

## Project structure
The project is divided into **two main components**:

### 1. Emotion Recognition Module  
Building an emotion detection system to make the QT robot more expressive and context-aware.

**Related Repositories**:
- [AffectiveTRM](https://github.com/Juste-Leo2/AffectiveTRM): Previously used for multimodal emotion recognition.  
  > **Note**: This repository is not used in the final pipeline as results were inconclusive, but is kept for reference.

**Progress**:  
- [x] Dataset collection  
- [x] Model training  
- [x] Model testing   

---

### 2. Full Low-Resource Interaction Pipeline  
An offline, low-compute pipeline enabling real-time interaction on embedded hardware.

**Pipeline components**:
- [x] **LLM backend ([llama.cpp](https://github.com/ggml-org/llama.cpp))**  
- [x] **Speech-to-text ([Vosk](https://github.com/alphacep/vosk-api))**
- [x] **Text-to-speech ([Piper](https://github.com/OHF-Voice/piper1-gpl))**
- [x] **Custom agent construction with personalized prompts**  
- [x] **Unified orchestration layer (New "Executor" Thread Architecture)**  

---

## Features Under Development & Roadmap

- [x] Windows & Linux (x64) support using GitHub workflows
- [x] Script tested directly on the QT robot
- [x] **New Architecture**: Threaded executor with queue-based action management for thread safety.
- [x] **Connected Jacket**: Support via `--JKT` flag using [QT-Touch](https://github.com/Juste-Leo2/QT-Touch).
- [x] **API Support**: Optional Google Gemini API integration via `--API`.
- [x] **Headless Mode**: Run without UI using `--no-ui`.
- [x] `--QT` argument in `main.py`: to use when running on the robot.
- [x] `--pytest` argument in `main.py`: used to test the pipeline (excluding the UI).
- [x] `--ci-mode` argument in `download_model.py`: replaces the large 8B model with a smaller 350M model.
- [x] `--name` argument: Define custom robot wake-word.
- [ ] **Language Switching**: Ability to switch the system language between French and English.
- [ ] **Voice Selection**: Option to change the TTS voice gender (Male/Female).
- [ ] **OpenAI API Integration**: Extending `--API` support to include OpenAI models (currently restricted to Google models).
- [ ] **User Documentation**: Create a comprehensive guide/wiki to facilitate repository usage and setup for new users.

## Why make it public?
Building in the open for **transparency, feedback, and accountability**.  
**Contributors welcome** — feel free to open an issue if you want to help!