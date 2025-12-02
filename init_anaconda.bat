:: Install ipykernel inside the venv (only once)
python -m pip install ipykernel

:: Add it as a Jupyter kernel
python -m ipykernel install --user --name 5AT020 --display-name "Python (5AT020)"

:: Download eDrives repository
git clone https://github.com/jdjotad/5AT020-eDrives temp

:: Copy the content of the repository to the current directory (rsync equivalent)
robocopy temp . /E /MOVE /XD .git /XF .gitignore

:: Force remove temp folder if it's still there (sometimes robocopy leaves it)
rmdir /s /q temp

:: Install the required packages
python -m pip install -r requirements.txt