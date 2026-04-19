import time
import pygame
import ollama
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoProcessor
import soundfile as sf
import torch
import os

# === CONFIG ===
MAX_SESSION_MINUTES = 45
VOICE_DESC = '<description="Friendly young girl voice named Mavi, soft playful warm American accent, excited and encouraging tone">'

# Load models once
print("Loading Mavi... (first time takes a minute)")
whisper = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
ollama.pull("llama3.2:3b")  # tiny & safe

# Maya1 (download happens automatically first time)
model = AutoModelForCausalLM.from_pretrained("maya-research/maya1", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("maya-research/maya1")

pygame.init()
screen = pygame.display.set_mode((800, 600))
avatar = pygame.image.load("assets/mavi_avatar.png") if os.path.exists("assets/mavi_avatar.png") else None

session_start = time.time()

def speak(text):
    # Add emotions for fun
    if "!" in text or "amazing" in text.lower():
        text = text.replace("!", " <excited>!")
    prompt = f"{VOICE_DESC} {text}"
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500)
    audio = processor.decode(output[0])
    sf.write("mavi_reply.wav", audio, 24000)
    pygame.mixer.music.load("mavi_reply.wav")
    pygame.mixer.music.play()
    # Simple mouth animation while speaking
    print("Mavi says:", text)

def listen():
    # Record 5 seconds or until silence (simple version - can improve)
    print("Listening... (say something to Mavi!)")
    # For real mic input you can add sounddevice later
    # Placeholder - in practice use a short recording
    # ... (full mic code in next version if you want)
    return "Hello Mavi"  # replace with actual whisper transcription

while True:
    if time.time() - session_start > MAX_SESSION_MINUTES * 60:
        speak("Wow we had the BEST time today! But it's time for a break. Let's go read a real book or play outside. I love you so much! See you tomorrow!")
        break

    user_text = listen()  # your daughter's spoken words
    if not user_text:
        continue

    # Send to brain with full prompt
    full_prompt = open("mavi_system_prompt.txt").read() + f"\nGirl: {user_text}\nMavi:"
    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": full_prompt}])['message']['content']

    speak(response)

    # Game triggers (example)
    if "this is" in user_text.lower():
        speak("This is... a cat! Can you spell C-A-T?")
    elif "animal" in user_text.lower():
        speak("I'm thinking of an animal that has stripes and says ROAR! What do you think it is?")