@echo off
REM Activate the skl_legacy conda environment
call %USERPROFILE%\miniconda3\condabin\conda.bat activate skl_legacy
echo.
echo Environment activated! You can now run:
echo   python scripts\run_sequence_svm.py
echo.
