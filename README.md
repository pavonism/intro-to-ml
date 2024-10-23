# Intercom Audio Classifier - Intro to ML course project 
The goal of the project is to prepare a machine learning module that can be hypothetically
used in an automated, voice-based intercom device.

# Setup

```console
$ python -m venv venv
$ venv/bin/activate (Linux) ./venv/Scripts/activate (Win)
$ pip install -e .
``` 

Additionally, you have to install Torch for cuda by yourself:

```console 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

# Run 

```console
$ python frontend
``` 

## 🌳 Project structure
```bash
│   .gitignore # Files to ignore
│   README.md # This file
│   pyproject.toml # Project configuration file
├───data
├───scripts # Notebooks
├───frontend # Basic frontend application 
└───backend # All application logic
```