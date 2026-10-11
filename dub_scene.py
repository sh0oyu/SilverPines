import os
import json
import torch

# Fix for PyTorch 2.6+ weights_only=True default on Google Colab
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    torch.serialization.add_safe_globals([XttsConfig])
except Exception:
    pass

try:
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([BaseDatasetConfig])
except Exception:
    pass

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

sucessos = 0
for idx, item in enumerate(dialogues, 1):
    line_id = item["id"]
    char = item["character"]
    text = item["text"]
    speaker_ref = item.get("speaker_ref", "references/marcio_simoes_ref.wav")
    
    out_file = os.path.join(out_dir, f"{line_id}_{char.replace(' ', '_')}.wav")
    print(f"[{idx}/{len(dialogues)}] Sintetizando {char}:")
    print(f"  FALA: \"{text}\"")
    
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_ref,
            language="pt",
            file_path=out_file,
            speed=1.0
        )
        print(f"  -> Gerado com sucesso: {out_file}")
        sucessos += 1
    except Exception as e:
        print(f"  ERRO ao gerar fala {line_id}: {e}")
    print()

print("=" * 60)
print(f"CONCLUIDO! {sucessos}/{len(dialogues)} falas foram sintetizadas.")
print("=" * 60)
