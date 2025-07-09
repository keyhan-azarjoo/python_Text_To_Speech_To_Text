# python_Text_To_Speech_To_Text


python -m venv myenv
myenv\Scripts\activate 
pip install openai-whisper  
pip install gTTS pydub langdetect
myenv\Scripts\activate  
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
pip install speechrecognition pyaudio 
pip install ipykernel
python -m ipykernel install --user --name myenv --display-name "Whisper Env"
pip install "spyder-kernels>=2.5,<2.6" 
