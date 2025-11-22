@echo off
setlocal

REM Verifie si le dossier de l'environnement virtuel existe.
REM S'il n'existe pas, l'installation n'a probablement pas ete faite.
set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if not exist ".venv" (
    echo Erreur: L'environnement Python n'a pas ete trouve.
    echo Veuillez d'abord executer 'setup.bat' pour effectuer l'installation.
    pause
    exit /b 1
)

echo Activation de l'environnement Python...
call .\.venv\Scripts\activate.bat

echo Lancement...
"%UV_EXE%" run src/download_model.py
"%UV_EXE%" run main.py

echo Fermeture...
call .\.venv\Scripts\deactivate.bat