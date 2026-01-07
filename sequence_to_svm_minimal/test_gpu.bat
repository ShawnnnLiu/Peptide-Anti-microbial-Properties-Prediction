@echo off
REM Test RTX 5070 GPU with ESMFold
echo.
echo Activating esm_env...
call %USERPROFILE%\miniconda3\condabin\conda.bat activate esm_env

echo.
echo Running GPU test...
python models\test_gpu_esmfold.py

pause
