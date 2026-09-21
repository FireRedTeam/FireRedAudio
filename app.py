"""FireRedAudio — unified audio understanding + generation Gradio demo.

Wraps `inference.FireRedAudioInference` from the repo root verbatim, so every task
uses the authors' own prompt templates and sampling parameters. Four tabs cover
zero-shot TTS, instruct TTS, speech editing (semantic / acoustic), and audio
understanding. The understanding tab exposes the full Qwen3 sampling controls
(`temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `do_sample`)
with a think-preset toggle that swaps in Qwen3's recommended `(temperature, top_p)`
for reasoning / non-reasoning modes.
"""

import argparse
import gc
import logging
import os
import random
import sys
import tempfile
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
import torchaudio
import gradio as gr

# ----------------------------------------------------------------- path bootstrap
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
ASSETS = REPO_ROOT / "assets" / "examples"

import inference as fra  # noqa: E402
from fireredaudio.redae.decoder import (  # noqa: E402
    PretrainedRedAEAudioDecoderV1,
    PretrainedRedAEDecoderConfig,
)


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("demo")
for _noisy in ("httpcore", "httpx", "asyncio", "urllib3", "uvicorn.access"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


MAX_SEED = np.iinfo(np.int32).max

# One inference lock: FireRedAudioInference holds a single model instance and
# every task path mutates the same `_gen_config` when we override sampling.
INFER_LOCK = threading.Lock()

# Populated in main()
ENGINE: "fra.FireRedAudioInference | None" = None


# ---------------------------------------------------------------- weight loading
def _build_engine(model_path: str, vae_decoder_path: str, device: str
                  ) -> "fra.FireRedAudioInference":
    """Build FireRedAudioInference on `device` and load the RedAE decoder.

    We load the decoder ourselves rather than passing `vae_decoder_path=` so we
    can drop the .pt file from disk before the (much larger) backbone lands —
    matches the HF Space's disk-frugal ordering.
    """
    logger.info("Loading RedAE decoder from %s", vae_decoder_path)
    vae_decoder = PretrainedRedAEAudioDecoderV1.from_config(
        PretrainedRedAEDecoderConfig()
    )
    sd = torch.load(vae_decoder_path, weights_only=True,
                    map_location="cpu", mmap=True)["model"]
    vae_decoder.load_state_dict(
        {k.removeprefix("decoder."): v for k, v in sd.items()
         if k.startswith("decoder.")},
        strict=True,
    )
    vae_decoder.eval()
    del sd
    gc.collect()

    logger.info("Loading FireRedAudio backbone from %s on %s", model_path, device)
    engine = fra.FireRedAudioInference(model_path=model_path, device=device)
    engine.vae_decoder = vae_decoder.to(engine.device)
    logger.info("Model ready.")
    return engine


# ---------------------------------------------------------------------- utilities
def _write_wav(audio: torch.Tensor) -> str:
    wav = audio.detach().float().cpu().reshape(1, -1).clamp(-1.0, 1.0)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    torchaudio.save(path, wav, fra.GENERATION_SAMPLE_RATE)
    return path


def _seed(seed: int, randomize: bool) -> int:
    seed = random.randint(0, MAX_SEED) if randomize else int(seed) % (MAX_SEED + 1)
    fra.set_seed(seed)
    return seed


# ----------------------------------------------------------------------- handlers
def clone_voice(
    reference_audio: str,
    reference_text: str,
    target_text: str,
    language: str = "zh",
    seed: int = 0,
    randomize_seed: bool = True,
    inference_cfg: float = 2.0,
    n_timesteps: int = 10,
    max_new_audio_steps: int = 150,
) -> tuple[str, int]:
    """Zero-shot voice cloning."""
    if not reference_audio:
        raise gr.Error("Please provide a reference audio clip.")
    if not (reference_text or "").strip():
        raise gr.Error("Please provide the transcript of the reference audio.")
    if not (target_text or "").strip():
        raise gr.Error("Please provide the text to synthesise.")
    with INFER_LOCK:
        used = _seed(seed, randomize_seed)
        out = ENGINE.tts(
            prompt_text=reference_text.strip(),
            prompt_audio=reference_audio,
            target_text=target_text.strip(),
            language=language,
            n_timesteps=int(n_timesteps),
            inference_cfg=float(inference_cfg),
            max_new_audio_steps=int(max_new_audio_steps),
        )
    return _write_wav(out.audio), used


def design_voice(
    instruction: str,
    text: str,
    seed: int = 0,
    randomize_seed: bool = True,
    inference_cfg: float = 2.0,
    n_timesteps: int = 10,
    max_new_audio_steps: int = 150,
    max_new_text_tokens: int = 512,
) -> tuple[str, str, int]:
    """Instruct TTS: describe a voice, then speak text in it."""
    if not (instruction or "").strip():
        raise gr.Error("Please describe the voice you want.")
    if not (text or "").strip():
        raise gr.Error("Please provide the text to synthesise.")
    with INFER_LOCK:
        used = _seed(seed, randomize_seed)
        out = ENGINE.voice_design(
            instruction=instruction.strip(),
            text=text.strip(),
            n_timesteps=int(n_timesteps),
            inference_cfg=float(inference_cfg),
            max_new_audio_steps=int(max_new_audio_steps),
            max_new_text_tokens=int(max_new_text_tokens),
        )
    return _write_wav(out.audio), out.text or "", used


def edit_speech(
    audio: str,
    instruction: str,
    edit_type: str = "semantic",
    seed: int = 0,
    randomize_seed: bool = True,
    inference_cfg: float = 2.0,
    n_timesteps: int = 10,
    max_new_audio_steps: int = 150,
    max_new_text_tokens: int = 512,
) -> tuple[str, str, int]:
    """Semantic / acoustic speech edit, keeping the original voice."""
    if not audio:
        raise gr.Error("Please provide an audio clip to edit.")
    if not (instruction or "").strip():
        raise gr.Error("Please provide an edit instruction.")
    with INFER_LOCK:
        used = _seed(seed, randomize_seed)
        out = ENGINE.edit(
            audio_path=audio,
            instruction=instruction.strip(),
            edit_type=edit_type,
            n_timesteps=int(n_timesteps),
            inference_cfg=float(inference_cfg),
            max_new_audio_steps=int(max_new_audio_steps),
            max_new_text_tokens=int(max_new_text_tokens),
        )
    return _write_wav(out.audio), out.text or "", used


# Qwen3-recommended sampling per mode: only (temperature, top_p) differ. Toggling
# `enable_thinking` in the UI re-applies these — a manual tune is overwritten,
# which is intended (predictable > sticky).
_THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95}
_NONTHINK_SAMPLING = {"temperature": 0.7, "top_p": 0.8}


def _apply_think_preset(enable_thinking: bool):
    preset = _THINK_SAMPLING if enable_thinking else _NONTHINK_SAMPLING
    return gr.update(value=preset["temperature"]), gr.update(value=preset["top_p"])


def understand_audio(
    audio: str,
    question: str,
    task: str,
    enable_thinking: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
) -> tuple[str, str]:
    """ASR / free-form audio QA with full Qwen3 sampling exposed.

    The upstream engine.understand() builds its GenerationConfig from a fixed
    per-task dict, so we override `engine._gen_config` for the duration of the
    call to inject the sliders' values without patching the packaged code.
    """
    if not audio:
        raise gr.Error("Please provide an audio clip.")
    prompt = fra.DEFAULT_ASR_PROMPT if task == "asr" else (question or "").strip()
    if not prompt:
        raise gr.Error("Please ask a question about the audio.")

    thinking = bool(enable_thinking) and task == "understand"

    # asr goes through beam search — sampling knobs would be silently ignored by
    # HF's GenerationConfig.validate() and just spam warnings. Only apply the
    # override for `understand`.
    overrides: dict = {}
    if task == "understand":
        overrides = {
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            "repetition_penalty": float(repetition_penalty),
        }
        # Greedy: neutralise sampling knobs so GenerationConfig.validate() doesn't
        # warn about "non-neutral value X while do_sample=False".
        if not overrides["do_sample"]:
            overrides.update(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0)

    max_new = fra.THINKING_MAX_NEW_TOKENS if thinking else int(max_new_tokens)

    with INFER_LOCK:
        original_gen_config = ENGINE._gen_config

        def _patched(t: str, mnt: int | None = None):
            cfg = original_gen_config(t, mnt)
            for k, v in overrides.items():
                setattr(cfg, k, v)
            return cfg

        ENGINE._gen_config = _patched
        try:
            out = ENGINE.understand(
                audio_paths=audio,
                prompt=prompt,
                task=task,
                enable_thinking=thinking,
                max_new_tokens=max_new,
            )
        finally:
            ENGINE._gen_config = original_gen_config

    return out.answer, out.reasoning or ""


# --------------------------------------------------------------------------- UI
_XHS_RED   = "#ff2442"   # Xiaohongshu brand red
_XHS_RED_H = "#ff4d63"
_FIRE      = "#e8593c"   # complementary FireRed accent

_THEME = (
    gr.themes.Base(
        primary_hue=gr.themes.colors.red,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        font=[
            "ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont",
            "Segoe UI", "PingFang SC", "Hiragino Sans GB",
            "Microsoft YaHei", "sans-serif",
        ],
    )
    .set(
        body_background_fill_dark="#0f1216",
        background_fill_primary_dark="#171b22",
        background_fill_secondary_dark="#1a1f2c",
        block_background_fill_dark="#171b22",
        border_color_primary_dark="#2a3140",
        block_border_color_dark="#2a3140",
        block_border_width="1px",
        block_border_width_dark="1px",
        body_text_color_dark="#e6ebf2",
        body_text_color_subdued_dark="#93a0b5",
        block_label_text_color_dark="#93a0b5",
        block_label_text_size="*text_sm",
        color_accent=_XHS_RED,
        border_color_accent_dark=_XHS_RED,
        button_primary_background_fill=_XHS_RED,
        button_primary_background_fill_dark=_XHS_RED,
        button_primary_background_fill_hover=_XHS_RED_H,
        button_primary_background_fill_hover_dark=_XHS_RED_H,
        button_primary_border_color=_XHS_RED,
        button_primary_border_color_dark=_XHS_RED,
        button_primary_border_color_hover=_XHS_RED_H,
        button_primary_border_color_hover_dark=_XHS_RED_H,
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_shadow="none",
        button_primary_shadow_dark="none",
        button_large_text_size="*text_md",
        button_large_padding="12px 28px",
        input_background_fill_dark="#1a1f2c",
        input_border_color_dark="#2a3140",
        input_border_color_focus_dark=_XHS_RED,
        block_radius="12px",
        input_radius="10px",
        button_large_radius="10px",
    )
)

_CSS = f"""
:root {{ color-scheme: dark; }}
#col-container {{ margin: 0 auto; max-width: 1180px; }}

/* header banner — XHS-red → FireRed gradient */
.app-header {{
    background: linear-gradient(135deg, {_XHS_RED} 0%, {_FIRE} 100%);
    border: none;
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 6px;
    box-shadow: 0 8px 24px rgba(255, 36, 66, 0.15);
    display: flex; align-items: center; gap: 18px;
}}
.app-header .brand-logo {{
    /* Transparent so the FireRedTeam avatar (already red-on-red) shows its
       own baked-in rounded shape; a white ring lifts it off the red gradient. */
    width: 60px; height: 60px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; overflow: hidden;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.85),
                0 4px 12px rgba(0,0,0,0.18);
}}
.app-header .brand-logo img {{
    width: 100%; height: 100%; object-fit: cover;
    display: block;
}}
.app-header .brand-text h1 {{
    font-size: 24px !important; font-weight: 800 !important;
    margin: 0 0 4px !important; color: #ffffff !important;
    letter-spacing: -0.3px;
}}
.app-header .brand-text p {{
    margin: 0 !important; color: rgba(255,255,255,0.9) !important;
    font-size: 13.5px !important;
}}
.app-header .brand-text .xhs-chip {{
    display: inline-block; margin-top: 8px;
    background: rgba(255,255,255,0.18); color: #ffffff;
    padding: 3px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
    backdrop-filter: blur(6px);
}}

/* tab underline in XHS red */
.tab-nav button.selected {{
    border-bottom-color: {_XHS_RED} !important;
    color: {_XHS_RED} !important;
}}

/* footer credit */
.app-footer {{
    text-align: center; color: #93a0b5;
    font-size: 12px; padding: 16px 0 4px;
}}
.app-footer a {{ color: {_XHS_RED}; text-decoration: none; }}

/* result / reasoning textboxes */
#think-box textarea {{
    font-family: "SF Mono", "Consolas", "Monaco", monospace !important;
    font-size: 12.5px !important; line-height: 1.65 !important;
    color: #93a0b5 !important; font-style: italic;
    background: #141822 !important; min-height: 120px;
}}
#think-box .block-label {{ color: #b48a5d !important; }}
#answer-box textarea {{
    font-size: 14px !important; line-height: 1.7 !important;
    min-height: 220px;
}}

details summary {{ font-size: 13px !important; font-weight: 500 !important; }}
"""

_DARK_JS = "() => { document.documentElement.classList.add('dark'); }"

_HEADER_HTML = """
<div class="app-header">
  <div class="brand-logo">
    <img src="gradio_api/file={logo_path}" alt="FireRedAudio" />
  </div>
  <div class="brand-text">
    <h1>🔥 FireRedAudio · Unified Audio Understanding & Generation</h1>
    <p>Zero-shot TTS · Instruct TTS · speech editing · ASR & audio QA — all in one 9B model.</p>
    <span class="xhs-chip">🔴 小红书 · FireRedTeam</span>
  </div>
</div>
"""

_FOOTER_HTML = """
<div class="app-footer">
  Built with ❤️ by <a href="https://github.com/FireRedTeam" target="_blank">FireRedTeam · 小红书 (Xiaohongshu)</a>
  · <a href="https://github.com/FireRedTeam/FireRedAudio" target="_blank">GitHub</a>
  · <a href="https://huggingface.co/FireRedTeam/FireRedAudio" target="_blank">🤗 Model</a>
  · <a href="https://arxiv.org/abs/2608.24168" target="_blank">📄 Paper</a>
  · <a href="https://fireredteam.github.io/demos/fireredaudio/" target="_blank">🔊 Samples</a>
</div>
"""


def build_demo() -> gr.Blocks:
    # Use the FireRedTeam org avatar (square, red-on-white wordmark) rather than
    # `fireredaudio_logo.png`, which is a 16:9 feature-infographic banner and
    # renders as unreadable colour mush when squeezed into a header icon slot.
    logo_path = REPO_ROOT / "assets" / "fireredteam_avatar.png"
    # Gradio 6.0 moved theme / css / js out of Blocks() into launch(); pass them
    # there instead. Blocks() only keeps `title` and layout kwargs.
    with gr.Blocks(title="FireRedAudio · 小红书 FireRedTeam") as demo:
        with gr.Column(elem_id="col-container"):
            gr.HTML(_HEADER_HTML.format(logo_path=str(logo_path)))

            # ------------------------------------------------------ Zero-shot TTS
            with gr.Tab("🗣️  Zero-shot TTS"):
                gr.Markdown(
                    "Give a few seconds of a voice plus its exact transcript, "
                    "and read any new text in that voice (zero-shot TTS)."
                )
                with gr.Row():
                    with gr.Column():
                        tts_ref_audio = gr.Audio(label="Reference audio",
                                                 type="filepath",
                                                 sources=["upload", "microphone"])
                        tts_ref_text = gr.Textbox(
                            label="Reference transcript",
                            placeholder="Exactly what is said in the reference audio",
                            lines=2,
                        )
                        tts_target = gr.Textbox(
                            label="Text to speak",
                            placeholder="What the cloned voice should say",
                            lines=3,
                        )
                        tts_lang = gr.Radio(["zh", "en"], value="zh",
                                            label="Language of the text")
                        tts_btn = gr.Button("Generate speech", variant="primary")
                    with gr.Column():
                        tts_out = gr.Audio(label="Generated speech", type="filepath")
                        tts_seed_out = gr.Number(label="Seed used",
                                                 precision=0, interactive=False)
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        tts_seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
                        tts_rand = gr.Checkbox(value=True, label="Randomize seed")
                    with gr.Row():
                        tts_cfg = gr.Slider(1.0, 5.0, value=2.0, step=0.1,
                                            label="CFG scale")
                        tts_steps = gr.Slider(4, 32, value=10, step=1,
                                              label="Flow-matching steps")
                        tts_max_audio = gr.Slider(
                            24, 750, value=150, step=1,
                            label="Max audio steps (160 ms each)",
                        )
                gr.Examples(
                    examples=[
                        [str(ASSETS / "tts_zh_prompt.wav"),
                         "同时，他强调微调要科学有序。",
                         "安徽淮南秦师傅发现，停在小区的爱车右前驾驶窗玻璃被砸。",
                         "zh"],
                    ],
                    inputs=[tts_ref_audio, tts_ref_text, tts_target, tts_lang],
                    outputs=[tts_out, tts_seed_out],
                    fn=clone_voice, cache_examples=False,
                    label="Official examples",
                )

            # -------------------------------------------------------- Instruct TTS
            with gr.Tab("🎨  Instruct TTS"):
                gr.Markdown(
                    "Describe a voice in words — gender, age, accent, emotion, "
                    "pacing — and the model invents it, then speaks your text."
                )
                with gr.Row():
                    with gr.Column():
                        vd_instruction = gr.Textbox(
                            label="Voice description",
                            placeholder="A warm middle-aged male voice with a slight British accent, speaking slowly.",
                            lines=4,
                        )
                        vd_text = gr.Textbox(label="Text to speak", lines=3)
                        vd_btn = gr.Button("Generate speech", variant="primary")
                    with gr.Column():
                        vd_out = gr.Audio(label="Generated speech", type="filepath")
                        vd_tags = gr.Textbox(label="Timbre tags chosen by the model",
                                             lines=3, interactive=False)
                        vd_seed_out = gr.Number(label="Seed used",
                                                precision=0, interactive=False)
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        vd_seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
                        vd_rand = gr.Checkbox(value=True, label="Randomize seed")
                    with gr.Row():
                        vd_cfg = gr.Slider(1.0, 5.0, value=2.0, step=0.1,
                                           label="CFG scale")
                        vd_steps = gr.Slider(4, 32, value=10, step=1,
                                             label="Flow-matching steps")
                        vd_max_audio = gr.Slider(
                            24, 750, value=150, step=1,
                            label="Max audio steps (160 ms each)",
                        )
                    vd_max_text = gr.Slider(
                        64, 1024, value=512, step=8,
                        label="Max text tokens (timbre tags)",
                    )
                gr.Examples(
                    examples=[
                        ["以女性高音区的清亮音色,表现出青年阶段的特质,音量略强,语速适中稍快,语调带有解释意味和急切的情感流露,确保语音流畅自然。",
                         "是我请他来的，可他什么也不知道，他来只是想打听一下，你们厂是不是有旧锅炉？"],
                        ["Incorporate an accent reminiscent of British English, perhaps "
                         "with a regional flavor such as Cockney, and convey a sense of "
                         "emotional vulnerability through a voice that reflects sadness "
                         "and an overwhelming demeanor, punctuated by a tremor.",
                         "I get these headaches, sharp pains through me head, milady. "
                         "Everything seems to be on top of me and I can't stop crying."],
                        ["Give it a dynamic tour as if you're a cheerful cartoon "
                         "character with a mid-to-high pitch broadcasting lively, "
                         "fast-paced thoughts.",
                         "Hello, everybody. Welcome to the show. My name is Ryan "
                         "Seacrest, and on behalf of the show, thank you for being here."],
                    ],
                    inputs=[vd_instruction, vd_text],
                    outputs=[vd_out, vd_tags, vd_seed_out],
                    fn=design_voice, cache_examples=False,
                    label="Official examples",
                )

            # ------------------------------------------------------- Speech editing
            with gr.Tab("✂️  Speech editing"):
                gr.Markdown(
                    "Edit a recording with a text instruction while keeping the "
                    "original voice. The model is trained on these exact instruction "
                    "templates rather than free-form phrasing.\n\n"
                    "**Semantic** — changes what is said:\n"
                    "- `delete 'the words to remove'.`\n"
                    "- `substitute 'old words' with 'new words'.`\n"
                    "- `substitute the characters or words from index 8 to index 10 with 'new words'.`\n"
                    "- `insert 'new words' after the character or word at index 8.`\n"
                    "- `insert 'new words' before the character or word 'anchor'.`\n\n"
                    "**Acoustic** — changes how it is said:\n"
                    "- `shifts the pitch by N steps.` — N in −6…−1, 1…6\n"
                    "- `adjusts the speed to X.` — X in 0.5…2.0\n"
                    "- `adjusts the volume to X.` — X in 0.3…2.0"
                )
                with gr.Row():
                    with gr.Column():
                        ed_audio = gr.Audio(label="Audio to edit", type="filepath",
                                            sources=["upload", "microphone"])
                        ed_instruction = gr.Textbox(label="Edit instruction", lines=2)
                        ed_type = gr.Radio(["semantic", "acoustic"], value="semantic",
                                           label="Edit type")
                        ed_btn = gr.Button("Apply edit", variant="primary")
                    with gr.Column():
                        ed_out = gr.Audio(label="Edited speech", type="filepath")
                        ed_text = gr.Textbox(
                            label="Rewritten transcript (semantic edits)",
                            info="Best-effort text the model wrote before rendering "
                                 "audio — substituted words sometimes show as a gap "
                                 "here even though the audio is correct.",
                            lines=3, interactive=False,
                        )
                        ed_seed_out = gr.Number(label="Seed used", precision=0,
                                                interactive=False)
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        ed_seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
                        ed_rand = gr.Checkbox(value=True, label="Randomize seed")
                    with gr.Row():
                        ed_cfg = gr.Slider(1.0, 5.0, value=2.0, step=0.1,
                                           label="CFG scale")
                        ed_steps = gr.Slider(4, 32, value=10, step=1,
                                             label="Flow-matching steps")
                        ed_max_audio = gr.Slider(
                            24, 750, value=150, step=1,
                            label="Max audio steps (160 ms each)",
                        )
                    ed_max_text = gr.Slider(
                        64, 1024, value=512, step=8,
                        label="Max text tokens (rewritten transcript)",
                    )
                gr.Examples(
                    examples=[
                        [str(ASSETS / "edit_semantic_zh_ref.wav"),
                         "delete '比普通的茶叶要'.", "semantic"],
                        [str(ASSETS / "edit_acoustic_zh_ref.wav"),
                         "shifts the pitch by 3 steps.", "acoustic"],
                    ],
                    inputs=[ed_audio, ed_instruction, ed_type],
                    outputs=[ed_out, ed_text, ed_seed_out],
                    fn=edit_speech, cache_examples=False,
                    label="Official examples",
                )

            # ------------------------------------------------------- Understanding
            with gr.Tab("👂  Listen & understand"):
                gr.Markdown(
                    "The same model transcribes speech (ASR) and answers open "
                    "questions about any audio — speech, music or sound events."
                )
                with gr.Row():
                    with gr.Column():
                        un_audio = gr.Audio(label="Audio", type="filepath",
                                            sources=["upload", "microphone"])
                        un_task = gr.Radio(
                            ["asr", "understand"], value="asr", label="Task",
                            info="asr = verbatim transcript (beam search) · "
                                 "understand = free-form QA",
                        )
                        un_question = gr.Textbox(
                            label="Question", value=fra.DEFAULT_ASR_PROMPT,
                            lines=2, interactive=False,
                        )
                        un_think = gr.Checkbox(
                            value=False, visible=False,
                            label="🧠 Enable thinking (understand only)",
                        )
                        un_btn = gr.Button("Run", variant="primary")
                    with gr.Column():
                        un_answer = gr.Textbox(label="Answer", lines=10,
                                               elem_id="answer-box")
                        un_reasoning = gr.Textbox(
                            label="🧠 Reasoning", lines=8,
                            interactive=False, elem_id="think-box",
                        )

                with gr.Accordion("⚙️  Sampling parameters (understand only)",
                                  open=False):
                    un_max_tokens = gr.Slider(
                        64, 4096, value=1024, step=8, label="max_new_tokens",
                    )
                    un_do_sample = gr.Checkbox(value=True, label="do_sample")
                    with gr.Row():
                        un_temp = gr.Slider(
                            0.0, 2.0, value=_NONTHINK_SAMPLING["temperature"],
                            step=0.05, label="temperature",
                        )
                        un_top_p = gr.Slider(
                            0.0, 1.0, value=_NONTHINK_SAMPLING["top_p"],
                            step=0.05, label="top_p",
                        )
                    with gr.Row():
                        un_top_k = gr.Slider(0, 200, value=20, step=1, label="top_k")
                        un_min_p = gr.Slider(0.0, 1.0, value=0.0, step=0.01,
                                             label="min_p")
                    un_rep_pen = gr.Slider(
                        0.5, 2.0, value=1.0, step=0.05,
                        label="repetition_penalty",
                    )

                def _on_task_change(task: str):
                    if task == "asr":
                        return (
                            gr.update(value=fra.DEFAULT_ASR_PROMPT,
                                      interactive=False),
                            gr.update(visible=False, value=False),
                        )
                    return (
                        gr.update(value="Describe the audio in detail.",
                                  interactive=True),
                        gr.update(visible=True),
                    )

                un_task.change(_on_task_change, [un_task], [un_question, un_think])
                un_think.change(_apply_think_preset, [un_think],
                                [un_temp, un_top_p])

                gr.Examples(
                    examples=[
                        [str(ASSETS / "asr_zh_fleurs.wav"),
                         fra.DEFAULT_ASR_PROMPT, "asr"],
                        [str(ASSETS / "assets_mmau_test.wav"),
                         "What illness did Second speaker's friend suffer from?\n"
                         "(A) Progressive arthritis (B) Progressive cancer "
                         "(C) Acute pneumonia (D) Chronic heart disease",
                         "understand"],
                    ],
                    inputs=[un_audio, un_question, un_task],
                    outputs=[un_answer, un_reasoning],
                    fn=lambda a, q, t: understand_audio(
                        a, q, t, False, 1024, True,
                        _NONTHINK_SAMPLING["temperature"],
                        _NONTHINK_SAMPLING["top_p"], 20, 0.0, 1.0,
                    ),
                    cache_examples=False,
                    label="Official examples",
                )

            gr.HTML(_FOOTER_HTML)

        tts_btn.click(
            _handler_wrapper(clone_voice),
            [tts_ref_audio, tts_ref_text, tts_target, tts_lang, tts_seed,
             tts_rand, tts_cfg, tts_steps, tts_max_audio],
            [tts_out, tts_seed_out],
        )
        vd_btn.click(
            _handler_wrapper(design_voice),
            [vd_instruction, vd_text, vd_seed, vd_rand, vd_cfg, vd_steps,
             vd_max_audio, vd_max_text],
            [vd_out, vd_tags, vd_seed_out],
        )
        ed_btn.click(
            _handler_wrapper(edit_speech),
            [ed_audio, ed_instruction, ed_type, ed_seed, ed_rand, ed_cfg,
             ed_steps, ed_max_audio, ed_max_text],
            [ed_out, ed_text, ed_seed_out],
        )
        un_btn.click(
            _handler_wrapper(understand_audio),
            [un_audio, un_question, un_task, un_think, un_max_tokens,
             un_do_sample, un_temp, un_top_p, un_top_k, un_min_p, un_rep_pen],
            [un_answer, un_reasoning],
        )

    return demo


def _handler_wrapper(fn):
    """Catches unexpected exceptions and surfaces them as gr.Error with a
    traceback in the log — a bare exception in a click handler just spins the
    UI forever with no feedback."""
    def _wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except Exception as e:
            logger.exception("handler %s failed", fn.__name__)
            raise gr.Error(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
    return _wrapped


# --------------------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(description="FireRedAudio Gradio Web Demo")
    p.add_argument(
        "--model_path", type=str, required=True,
        help="Directory holding the FireRedAudio checkpoint "
             "(config.json + safetensors shards).",
    )
    p.add_argument(
        "--vae_decoder_path", type=str, required=True,
        help="Path to the RedAE decoder .pt file.",
    )
    p.add_argument("--tokenizer_path", type=str, default=None,
                   help="Defaults to --model_path.")
    p.add_argument("--processor_path", type=str, default=None,
                   help="Defaults to --model_path.")
    p.add_argument("--device", type=str, default=None,
                   help="cuda:N or cpu (defaults to cuda:0 if available).")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument(
        "--root-path", dest="root_path", type=str, default="",
        help="Root path prefix when running behind a reverse proxy (e.g. /myapp).",
    )
    p.add_argument(
        "--share", action="store_true",
        help="Launch a public gradio.live tunnel (requires internet).",
    )
    return p.parse_args()


def main():
    global ENGINE
    args = parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":")[1]))

    ENGINE = _build_engine(
        model_path=args.model_path,
        vae_decoder_path=args.vae_decoder_path,
        device=args.device,
    )
    # Optional override paths — FireRedAudioInference already accepts them at
    # __init__, but we build the engine with defaults for a leaner signature;
    # honour explicit overrides by re-loading the tokenizer/processor here.
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        ENGINE.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        ENGINE.tokenizer.padding_side = "left"
    if args.processor_path:
        from fireredaudio.audio_encoder.processor import FireRedAudioProcessor
        ENGINE.processor = FireRedAudioProcessor.from_pretrained(args.processor_path)

    demo = build_demo()
    demo.queue(max_size=20).launch(
        server_name=args.host,
        server_port=args.port,
        root_path=args.root_path,
        share=args.share,
        show_error=True,
        allowed_paths=[str(REPO_ROOT / "assets")],
        theme=_THEME,
        css=_CSS,
        js=_DARK_JS,
    )


if __name__ == "__main__":
    main()
