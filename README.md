# QTRobot-Brain (WIP – Very Alpha)

**Status**: In active development. Code is experimental, unstable, and mostly undocumented.  
**Goal**: Build a complete, low-resource pipeline for real-time human–machine interaction on the QT robot (fine-tuned Phi-3 / <100ms latency target).  
**License**: Apache 2.0

---

## Project structure
The project is divided into **two main components**:

---

### 1. Emotion Recognition Module  
Building an emotion detection system to make the QT robot more expressive and context-aware.

**Progress**:  
- [x] Dataset collection  
- [ ] Model training  
- [ ] Model testing  
- [ ] Integration into QT robot  

---

### 2. Full Low-Resource Interaction Pipeline  
An offline, low-compute pipeline enabling real-time interaction on embedded hardware.

**Pipeline components**:
- [x] **LLM backend (llama.cpp)** – https://github.com/ggml-org/llama.cpp  
- [x] **Speech-to-text (Vosk)** – https://github.com/alphacep/vosk-api  
- [x] **Text-to-speech (Piper)** – https://github.com/OHF-Voice/piper1-gpl  
- [ ] **Custom agent construction with personalized prompts**  
- [ ] **Unified orchestration layer (complete pipeline build)**  

---

## Why make it public?
Building in the open for **transparency, feedback, and accountability**.  
**Contributors welcome** — feel free to open an issue if you want to help!
