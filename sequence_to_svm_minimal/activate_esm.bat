@echo off
REM Activate the esm_env conda environment for ESM models
call %USERPROFILE%\miniconda3\condabin\conda.bat activate esm_env
echo.
echo ESM Environment activated! 
echo Python: 3.10, PyTorch 2.x, CUDA 12.1, ESM models ready
echo.
