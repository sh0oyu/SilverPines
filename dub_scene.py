import os
import sys
import json
import traceback
import torch
import soundfile as sf

# ============================================================
# COMPATIBILITY MONKEY-PATCHES FOR MODERN PYTORCH & TRANSFORMERS
# ============================================================

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

# 4. Patch de atencao para transformers 4.45+ (resolve 'NoneType' object has no attribute 'shape')
try:
    from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel
    
    _orig_gpt2_forward = GPT2InferenceModel.forward
    def _patched_gpt2_forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
            
        past_key_values = kwargs.get("past_key_values")
        if past_key_values is None and len(args) > 1:
            past_key_values = args[1]
            
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None and len(args) > 2:
            attention_mask = args[2]
            
        if attention_mask is None and input_ids is not None:
            prefix_len = getattr(self, "cached_prefix_emb", None)
            prefix_l = prefix_len.shape[1] if prefix_len is not None else 0
            if past_key_values is not None and len(past_key_values) > 0:
                past_len = past_key_values[0][0].shape[-2]
                total_len = prefix_l + past_len + input_ids.shape[1]
            else:
                total_len = input_ids.shape[1]
                
            mask = torch.ones((input_ids.shape[0], total_len), dtype=torch.long, device=input_ids.device)
            kwargs["attention_mask"] = mask
            if len(args) > 2:
                args = list(args)
                args[2] = mask
                args = tuple(args)
                
        return _orig_gpt2_forward(self, *args, **kwargs)
        
    GPT2InferenceModel.forward = _patched_gpt2_forward
    print("Patch de atencao transformers aplicado com sucesso!")
except Exception as e:
    print(f"Aviso no patch GPT2InferenceModel: {e}")

# ============================================================
# INICIO DA SINTESE DE VOZ COM MULTI-TAKES DE ESTUDIO
# ============================================================
from TTS.api import TTS

print("=" * 60)
print("SILVER PINES - DUBLAGEM MASTER PT-BR (MULTI-SAMPLE CLONING)")
print("=" * 60)

script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Diretorio base: {script_dir}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo detectado: {device.upper()}")
if device == "cpu":
    print("AVISO: Executando em CPU. Para velocidade maxima, certifique-se de ativar a GPU T4.")

print("\nCarregando modelo XTTS-v2...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
xtts_model = tts.synthesizer.tts_model
print("Modelo carregado com sucesso!\n")

# Coletar todos os arquivos de audio de referencia da pasta Vozes
walker_dir = os.path.join(script_dir, "references", "gutemberg")
velho_dir = os.path.join(script_dir, "references", "marcio")

walker_refs = [os.path.normpath(os.path.join(walker_dir, f)) for f in sorted(os.listdir(walker_dir)) if f.endswith(".wav")]
velho_refs = [os.path.normpath(os.path.join(velho_dir, f)) for f in sorted(os.listdir(velho_dir)) if f.endswith(".wav")]

print(f"Takes encontrados para Gutemberg Barros (Red Walker): {len(walker_refs)} arquivos")
for r in walker_refs:
    print(f"  - {os.path.basename(r)} ({os.path.getsize(r)} bytes)")

print(f"\nTakes encontrados para Marcio Simoes (O Velho): {len(velho_refs)} arquivos")
for r in velho_refs:
    print(f"  - {os.path.basename(r)} ({os.path.getsize(r)} bytes)")

print("\nPre-computando embeddings vocais compostos (multi-sample)...")
print("  -> Extraindo perfil vocal do Gutemberg Barros...")
walker_gpt_latents, walker_spk_emb = xtts_model.get_conditioning_latents(
    audio_path=walker_refs,
    gpt_cond_len=30,
    gpt_cond_chunk_len=6
)

print("  -> Extraindo perfil vocal do Marcio Simoes...")
velho_gpt_latents, velho_spk_emb = xtts_model.get_conditioning_latents(
    audio_path=velho_refs,
    gpt_cond_len=30,
    gpt_cond_chunk_len=6
)
print("Perfis vocais extraidos com perfeicao!\n")

out_dir = os.path.join(script_dir, "output_dublagem")
os.makedirs(out_dir, exist_ok=True)

dialogues_path = os.path.join(script_dir, "scene_dialogue.json")
with open(dialogues_path, "r", encoding="utf-8") as f:
    dialogues = json.load(f)

print(f"Total de falas a dublar: {len(dialogues)}\n")

sucessos = 0
for idx, item in enumerate(dialogues, 1):
    line_id = item["id"]
    char = item["character"]
    text = item["text"]
    
    out_file = os.path.join(out_dir, f"{line_id}_{char.replace(' ', '_')}.wav")
    print(f"[{idx}/{len(dialogues)}] Sintetizando {char}:")
    print(f"  FALA: \"{text}\"")
    
    gpt_latents, spk_emb = (walker_gpt_latents, walker_spk_emb) if char == "Red Walker" else (velho_gpt_latents, velho_spk_emb)
    
    try:
        out = xtts_model.inference(
            text=text,
            language="pt",
            gpt_cond_latent=gpt_latents,
            speaker_embedding=spk_emb,
            temperature=0.75,
            top_p=0.85,
            top_k=50,
            repetition_penalty=4.0,
            length_penalty=1.0,
            enable_text_splitting=False
        )
        sf.write(out_file, out["wav"], 24000)
        print(f"  -> Salvo com sucesso: {out_file}")
        sucessos += 1
    except Exception as e:
        print(f"  ERRO ao gerar fala {line_id}:")
        traceback.print_exc()
    print()

print("=" * 60)
print(f"CONCLUIDO! {sucessos}/{len(dialogues)} falas foram sintetizadas.")
print("=" * 60)
