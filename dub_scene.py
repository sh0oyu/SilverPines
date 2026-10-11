import os
import json
import torch

# Aceitar termos da licenca do Coqui automaticamente
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

print("=" * 60)
print("SILVER PINES - DUBLAGEM PT-BR COM XTTS-v2")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo detectado: {device.upper()}")
if device == "cpu":
    print("AVISO: Processando em CPU (mais lento). Para velocidade maxima, use GPU T4 no Colab.")

print("Carregando modelo XTTS-v2...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print("Modelo carregado com sucesso!")
print()

out_dir = "output_dublagem"
os.makedirs(out_dir, exist_ok=True)

with open("scene_dialogue.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

print(f"Total de falas na cena: {len(dialogues)}")
print()

for idx, item in enumerate(dialogues, 1):
    line_id = item["id"]
    char = item["character"]
    text = item["text"]
    speaker_ref = item.get("speaker_ref", "references/marcio_simoes_ref.wav")
    
    out_file = os.path.join(out_dir, f"{line_id}_{char.replace(' ', '_')}.wav")
    print(f"[{idx}/{len(dialogues)}] Sintetizando {char}:")
    print(f"  FALA: \"{text}\"")
    
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_ref,
        language="pt",
        file_path=out_file,
        speed=1.0
    )
    print(f"  -> Gerado: {out_file}")
    print()

print("=" * 60)
print("CONCLUIDO! Todas as falas foram sintetizadas.")
print("=" * 60)
