@echo off
setlocal

:: =============================================================================
::      INITIALISATION DE L'ENVIRONNEMENT DE DEVELOPPEMENT
:: =============================================================================
echo.

REM --- Etape 1: Verification des pre-requis ---
echo [1/5] Verification des pre-requis...
if not exist "requirements.txt" (
    echo ECHEC: Le fichier 'requirements.txt' est introuvable.
    goto:error
)
echo    [OK] Fichier requirements.txt trouve.

REM --- Etape 2: Verification et installation de 'uv' ---
echo [2/5] Verification et installation de 'uv'...

REM On verifie si la commande 'uv' existe deja
where uv > nul 2>&1
if %errorlevel% == 0 (
    echo    [OK] uv est deja installe.
) else (
    echo    uv non trouve, lancement de l'installation...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo ECHEC: Impossible d'installer uv. Verifiez votre connexion internet ou les permissions PowerShell.
        goto:error
    )
    
    REM On verifie que l'installation a bien ajoute uv au PATH
    where uv > nul 2>&1
    if %errorlevel% neq 0 (
        echo ECHEC: uv a ete installe mais n'est pas accessible. Vous devrez peut-etre redemarrer votre terminal.
        goto:error
    )
    echo    [OK] uv installe avec succes.
)


REM --- Etape 3: Creation de l'environnement Python ---
echo [3/5] Creation de l'environnement Python (python 3.11)...
uv venv -p 3.11
if %errorlevel% neq 0 (
    echo ECHEC: Impossible de creer l'environnement Python. Verifiez que Python 3.11 est accessible.
    goto:error
)
echo    [OK] Environnement Python cree dans le dossier .venv.

REM --- Etape 4: Activation et installation des dependances ---
echo [4/5] Installation des dependances Python...

REM Verification que le script d'activation existe
if not exist ".\.venv\Scripts\activate.bat" (
    echo ECHEC: Le script d'activation de l'environnement n'a pas ete trouve.
    goto:error
)

call .\.venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ECHEC: Impossible d'activer l'environnement virtuel.
    goto:error
)

uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ECHEC: Impossible d'installer les dependances depuis requirements.txt.
    call .\.venv\Scripts\deactivate.bat
    goto:error
)

echo    [OK] Dependances Python installees.

REM --- Etape 5: Nettoyage et finalisation ---
echo [5/5] Finalisation de l'installation...
call .\.venv\Scripts\deactivate.bat
echo    [OK] Environnement desactive.
goto:success


:success
echo.
echo ==========================================================
echo      ENVIRONNEMENT CONFIGURE AVEC SUCCES !
echo ==========================================================
echo.
echo Pour activer l'environnement, executez la commande :
echo   call .\.venv\Scripts\activate.bat
echo.
goto:end


:error
echo.
echo ==========================================================
echo      ERREUR: L'INSTALLATION A ECHOUE.
echo ==========================================================
echo Veuillez verifier les messages d'erreur ci-dessus.
echo.
set "EXIT_CODE=1"
goto:end


:end
endlocal
echo Appuyez sur une touche pour quitter...
pause > nul
exit /b %EXIT_CODE%