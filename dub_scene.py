import os
import sys
import json
import traceback
import torch

# 1. Fix para transformers 4.45+ que removeu 'isin_mps_friendly'
try:
    import transformers.pytorch_utils
    if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
        transformers.pytorch_utils.isin_mps_friendly = torch.isin
except Exception:
    pass

# 2. Fix para PyTorch 2.6+ weights_only=True padrao no Colab
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

# 3. Aceitar termos da licenca do Coqui automaticamente
os.environ["COQUI_TOS_AGREED"] = "1"

# Importar TTS
from TTS.api import TTS

print("=" * 60)
print("SILVER PINES - DUBLAGEM PT-BR COM XTTS-v2")
print("=" * 60)

script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Diretorio base do script: {script_dir}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo detectado: {device.upper()}")
if device == "cpu":
    print("AVISO: Processando em CPU (mais lento). Para velocidade maxima, use GPU T4 no Colab.")

# Validar existencia dos arquivos de audio de referencia
for ref_file in ["references/gutemberg_walker_ref.wav", "references/marcio_simoes_ref.wav"]:
    full_path = os.path.normpath(os.path.join(script_dir, ref_file))
    if not os.path.exists(full_path):
        print(f"ERRO CRITICO: Arquivo de referencia nao encontrado: {full_path}")
    else:
        print(f"Referencia OK: {ref_file} ({os.path.getsize(full_path)} bytes)")

print("\nCarregando modelo XTTS-v2...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print("Modelo carregado com sucesso!\n")

out_dir = os.path.join(script_dir, "output_dublagem")
os.makedirs(out_dir, exist_ok=True)

dialogues_path = os.path.join(script_dir, "scene_dialogue.json")
with open(dialogues_path, "r", encoding="utf-8") as f:
    dialogues = json.load(f)

print(f"Total de falas na cena: {len(dialogues)}\n")

sucessos = 0
for idx, item in enumerate(dialogues, 1):
    line_id = item["id"]
    char = item["character"]
    text = item["text"]
    rel_speaker_ref = item.get("speaker_ref", "references/marcio_simoes_ref.wav")
    speaker_ref = os.path.normpath(os.path.join(script_dir, rel_speaker_ref))
    
    out_file = os.path.join(out_dir, f"{line_id}_{char.replace(' ', '_')}.wav")
    print(f"[{idx}/{len(dialogues)}] Sintetizando {char}:")
    print(f"  FALA: \"{text}\"")
    print(f"  REF:  {speaker_ref}")
    
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_ref,
            language="pt",
            file_path=out_file,
            split_sentences=False
        )
        print(f"  -> Gerado com sucesso: {out_file}")
        sucessos += 1
    except Exception as e:
        print(f"  ERRO ao gerar fala {line_id}:")
        traceback.print_exc()
    print()

print("=" * 60)
print(f"CONCLUIDO! {sucessos}/{len(dialogues)} falas foram sintetizadas.")
print("=" * 60)
