@echo off
echo Setting up Insect CNN Model Environment...
echo.

echo Activating virtual environment...
call .venv\Scripts\activate

echo.
echo Installing required packages...
pip install --upgrade pip
pip install tensorflow numpy matplotlib scikit-learn seaborn pillow

echo.
echo Testing installation...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import numpy as np; print('NumPy version:', np.__version__)"
python -c "import matplotlib; print('Matplotlib installed successfully')"
python -c "import sklearn; print('Scikit-learn installed successfully')"
python -c "import seaborn; print('Seaborn installed successfully')"
python -c "import PIL; print('Pillow installed successfully')"

echo.
echo Running setup test...
python test_setup.py

echo.
echo Setup complete! You can now run:
echo python insect_cnn_model.py
pause
