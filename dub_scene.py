import os
import json
import torch
from TTS.api import TTS

print("Checking CUDA availability...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

print("Loading XTTS-v2 model...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

out_dir = "output_dublagem"
os.makedirs(out_dir, exist_ok=True)

with open("scene_dialogue.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

print(f"Total lines to dub: {len(dialogues)}")

for item in dialogues:
    line_id = item["id"]
    char = item["character"]
    text = item["text"]
    speaker_ref = item.get("speaker_ref", "references/marcio_simoes_ref.wav")
    
    out_file = os.path.join(out_dir, f"{line_id}_{char.replace(' ', '_')}.wav")
    print(f"Generating [{line_id}] {char}: '{text}'...")
    
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_ref,
        language="pt",
        file_path=out_file,
        speed=0.95
    )
    print(f"  -> Saved {out_file}")

print("\nAll lines dubbed successfully!")
