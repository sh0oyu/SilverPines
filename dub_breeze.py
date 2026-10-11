import os
import sys
import json
import math
from pathlib import Path
import soundfile as sf

# Garantir que os modulos do breeze-tts sejam encontrados
sys.path.insert(0, "/content/breeze-tts")

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

print("=" * 60)
print("SILVER PINES - DUBLAGEM OFICIAL COM BREEZE TTS 2")
print("=" * 60)

model_path = Path("/content/breeze-tts-2")
if not model_path.exists():
    raise FileNotFoundError(f"Pasta do modelo nao encontrada em {model_path}. Execute o download do Hugging Face primeiro!")

print("Carregando modelo Breeze TTS 2 na GPU...")
device = resolve_device()
print(f"Dispositivo de execucao: {device}")

tokenizer, model, audio_tokenizer = load_runtime(
    model_path,
    device=device,
    attn_implementation="eager",
)
update_generation_config_for_breeze(model)

config = FastStreamingConfig(
    max_new_tokens=1500,
    max_seq_len=2048,
    fast_all=False,
    repetition_penalty=1.1,
)
runtime = FastBreezeStreamingRuntime(
    model, audio_tokenizer, config, tokenizer=tokenizer
)
print("Breeze TTS 2 carregado com sucesso!\n")

# Configuracao das referencias e direcao de voz (Voice Direction)
gutemberg_ref_audio = "/content/SilverPines/references/gutemberg/gutemberg_15.wav"
gutemberg_ref_text = "Silver Pines... este lugar parece congelado no tempo."
gutemberg_instruction = "A deep, gritty, cynical, hardened noir private detective voice with weary breathing and realistic pacing."

marcio_ref_audio = "/content/SilverPines/references/marcio/marcio_05.wav"
marcio_ref_text = "Ha! Você não sabe como achar ele, então quer que eu faça todo o trabalho sujo."
marcio_instruction = "A menacing, mocking, sneering, dangerous elderly psychopathic villain voice with a chilling edge."

out_dir = Path("/content/SilverPines/output_dublagem_breeze")
out_dir.mkdir(parents=True, exist_ok=True)

dialogues_path = "/content/SilverPines/scene_dialogue.json"
with open(dialogues_path, "r", encoding="utf-8") as f:
    dialogues = json.load(f)

print(f"Total de falas na cena: {len(dialogues)}\n")

for idx, item in enumerate(dialogues, 1):
    line_id = item["id"]
    char = item["character"]
    text = item["text"]
    
    out_file = out_dir / f"{line_id}_{char.replace(' ', '_')}.wav"
    print(f"[{idx}/{len(dialogues)}] Breeze TTS 2 sintetizando {char}:")
    print(f"  FALA: \"{text}\"")
    
    if char == "Red Walker":
        ref_audio = gutemberg_ref_audio
        ref_text = gutemberg_ref_text
        instruction = gutemberg_instruction
    else:
        ref_audio = marcio_ref_audio
        ref_text = marcio_ref_text
        instruction = marcio_instruction
        
    request = {
        "id": line_id,
        "text": text,
        "speaker": "S0",
        "instruction": instruction,
        "ref_audio_path": ref_audio,
        "ref_text": ref_text
    }
    
    template_name = select_template_name(request)
    set_all_seeds(42)
    inputs = prepare_inputs(
        tokenizer,
        audio_tokenizer,
        model,
        [request],
        get_template(template_name),
        guidance_scale=1.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None
    )
    
    with sf.SoundFile(
        out_file,
        mode="w",
        samplerate=runtime.sample_rate,
        channels=1,
        subtype="PCM_16"
    ) as f:
        for chunk in runtime.iter_audio_chunks(inputs, request_id=line_id, seed=42):
            f.write(chunk.audio)
            
    print(f"  -> Salvo: {out_file}\n")

print("=" * 60)
print("CONCLUIDO! Todas as falas foram dubladas com sucesso pelo Breeze TTS 2!")
print("=" * 60)
