@echo off

if not exist ".env" (
    python -m venv .env
)

call .env\Scripts\activate.bat

pip show opencv-python >nul 2>&1
if errorlevel 1 (
    echo Installing opencv-python...
    python -m pip install opencv-python
)

pip show pillow >nul 2>&1
if errorlevel 1 (
    echo Installing pillow...
    python -m pip install pillow
)

pip show scikit-learn >nul 2>&1
if errorlevel 1 (
    echo Installing scikit-learn...
    python -m pip install scikit-learn
)

cls

:start
set /p user_input="Enter the path to the image or folder with images [and the output folder path]: "

for /f "tokens=1,2" %%a in ("%user_input%") do (
    set input_path=%%a
    set output_path=%%b
)

if defined output_path (
    python main.py "%input_path%" "%output_path%"
) else (
    python main.py "%input_path%"
)

set /p repeat="Do you want to process more images? (y/n): "
if /i "%repeat%"=="y" goto start

deactivate