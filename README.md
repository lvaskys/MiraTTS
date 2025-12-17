# MiraTTS
A high quality extremely fast TTS model

```python
from mira.model import MiraTTS
mira_tts = MiraTTS('YatharthS/MiraTTS')

file = r"C:\Users\Nitin\Downloads\Linus%20Tech%20Tips.wav"
file = file.replace("C:\\Users\\Nitin\\Downloads\\", "/mnt/c/Users/Nitin/Downloads/")

context_tokens = mira_tts.encode_audio(file)
audio = mira_tts.generate("Alright, so have you ever heard of a little thing named text to speech? Well, it allows you to convert text into speech!... I know, that's super cool, isn't it?", context_tokens)
from IPython.display import Audio
Audio(audio, rate=48000)

file = r"C:\Users\Nitin\Downloads\emotional_jessica1.mp3"
file = file.replace("C:\\Users\\Nitin\\Downloads\\", "/mnt/c/Users/Nitin/Downloads/")

context_tokens = mira_tts.encode_audio(file)

audio = mira_tts.batch_generate(["Hey, what's up! I am feeling SO happy!", "Honestly, this is really interesting, isn't it?"], [context_tokens])
from IPython.display import Audio
Audio(audio, rate=48000)

```
