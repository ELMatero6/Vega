@echo off

:: Create a new Conda environment with Python 3.10
call conda create --name vega python==3.10 --yes

:: Activate the new Conda environment
call conda activate vega

:: Install PyTorch, torchvision, and torchaudio
call conda install pytorch torchvision torchaudio -c pytorch --yes

cd ./EssentialFiles
cd ..
call python -m pip install -r requirements.txt

echo Returning to the main directory...

echo Installing Model...
python downloadmodel.py

echo Installation Complete. Run Start.Bat
PAUSE