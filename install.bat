@echo off
echo Installing Indian Stock Analysis...

:: Create and activate virtual environment
python -m venv .venv
call .venv\Scripts\activate

:: Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Install the package in development mode
pip install -e .

echo.
echo Installation complete! You can now run the application using:
echo indian-stock-analysis run
echo.
pause
