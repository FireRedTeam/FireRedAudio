"""FireRedAudio inference. One script for every task, selected with --task.

    # speech recognition
    python inference.py --task asr --model <ckpt> --audio a.wav

    # audio understanding and QA; several --audio for e.g. speaker verification,
    # --enable-thinking to let the model reason first
    python inference.py --task understand --model <ckpt> --audio a.wav \\
        --prompt "What is the speaker's emotion?"

    # ICL voice cloning from a reference audio and its transcript
    python inference.py --task tts --model <ckpt> --vae-decoder <redae.pt> \\
        --prompt-audio ref.wav --prompt-text "<transcript of ref.wav>" \\
        --target-text "<text to synthesize>" --language zh --output out.wav

    # speech editing; semantic rewrites content, acoustic changes pitch/rate/volume
    python inference.py --task edit --model <ckpt> --vae-decoder <redae.pt> \\
        --audio a.wav --instruction "adjust the speed to 0.5" --edit-type acoustic

    # synthesis from a timbre description
    python inference.py --task voice_design --model <ckpt> --vae-decoder <redae.pt> \\
        --instruction "<timbre description>" --text "<text to synthesize>"

The model emits 25 Hz AE latents, so generation tasks need --vae-decoder pointing at
the RedAE weights to reach a waveform.

FireRedAudioInference holds the logic and can be imported directly; the CLI below is
a thin wrapper.
"""

import argparse
import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

from transformers import AutoTokenizer, GenerationConfig

from fireredaudio.audio_encoder.processor import FireRedAudioProcessor
from fireredaudio.data.prompt_encoder import (
    FEAT_TYPE_GENERATION,
    FEAT_TYPE_UNDERSTAND,
    AudioPromptEncoder,
)
from fireredaudio.loading import load_fireredaudio
from fireredaudio.utils.audio import (
    GENERATION_SAMPLE_RATE,
    UNDERSTAND_SAMPLE_RATE,
    read_audio,
)
from fireredaudio.redae.decoder import (
    PretrainedRedAEAudioDecoderV1,
    PretrainedRedAEDecoderConfig,
)
from fireredaudio.redae.encoder import pad_to_multiple_of

logger = logging.getLogger(__name__)

# ===================================================================== Constants

UNDERSTAND_TASKS = ("asr", "understand")
GENERATION_TASKS = ("tts", "edit", "voice_design")

# understand is the only task with CoT. asr uses beam search for a deterministic
# transcript, and its think block is always empty in training, as it is for the
# generation tasks.
THINKING_TASKS = ("understand",)

DEFAULT_ASR_PROMPT = "Transcribe speech to text."

GENERIC_SYSTEM_PROMPT = "You are a helpful assistant."
UNDERSTAND_SYSTEM_PROMPT = (
    "You are an audio understanding expert. Please answer user questions based on the audio."
)

# CoT needs room for a full reasoning pass before the answer; 300 truncates it
# mid-sentence, before even reaching </think>.
THINKING_MAX_NEW_TOKENS = 1024

# Text sampling. eos / pad ids are resolved from the tokenizer at runtime.
# max_new_tokens applies to understanding tasks only: generate_tts sizes its text
# span with max_new_text_tokens and ignores this field.
_TEXT_SAMPLING = dict(
    do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
    repetition_penalty=1.0, max_new_tokens=300,
)
GENERATION_PARAMS = {
    "asr": dict(do_sample=False, num_beams=4, repetition_penalty=1.1, max_new_tokens=300),
    "understand": dict(_TEXT_SAMPLING),
    "tts": dict(_TEXT_SAMPLING),
    "edit": dict(_TEXT_SAMPLING),
    "voice_design": dict(_TEXT_SAMPLING),
}


def set_seed(seed: int) -> None:
    """torch.randn in flow matching is the only randomness behind the audio output;
    random / numpy are seeded defensively."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# =============================================================== Prompt building
#
# All four tasks share one chatml skeleton and differ in three slots:
#
#   task          system             user                          assistant_prefix
#   ------------  -----------------  ----------------------------  ----------------------
#   understand    audio expert       "Audio N: ..." xN + question  empty
#   tts           generic assistant  "Convert text to speech.\n…"  <|sosp|> + placeholder
#   edit          generic assistant  "Audio 1: …\n{instruction}"   empty
#   voice_design  generic assistant  "{timbre}\n\n…\n{text}"       empty
#
# Only tts has a non-empty assistant_prefix, which makes generate_tts start in audio
# mode; the other three start in text mode and emit <|sosp|> themselves. The think
# block is empty unless enable_thinking leaves it open at "<think>\n" (see
# THINKING_TASKS). The templates must match training character for character.

def _chatml(system: str, user: str, assistant_prefix: str = "",
            enable_thinking: bool = False) -> str:
    if enable_thinking and assistant_prefix:
        # A tts prompt ends on <|sosp|> + placeholder to force audio mode, which is
        # incompatible with leaving the think block open.
        raise ValueError("enable_thinking cannot be combined with assistant_prefix")
    think = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{think}{assistant_prefix}"
    )


def build_understand_prompt(prompt: str, num_audios: int, audio_sp_token: str,
                            enable_thinking: bool = False) -> str:
    """ASR and audio understanding.

    Takes any number of audios, numbered "Audio N: " in order and placed before the
    question. Speaker verification, for instance, passes two.
    """
    audio_segs = "".join(
        f"Audio {i + 1}: <|sosp|>{audio_sp_token}<|eosp|>\n" for i in range(num_audios)
    )
    return _chatml(UNDERSTAND_SYSTEM_PROMPT, f"{audio_segs}{prompt}",
                   enable_thinking=enable_thinking)


def build_tts_prompt(prompt_text: str, target_text: str, language: str,
                     audio_sp_token: str) -> str:
    """ICL voice cloning. `en` inserts a space between prompt and target text."""
    sep = " " if language == "en" else ""
    return _chatml(
        GENERIC_SYSTEM_PROMPT,
        f"Convert text to speech.\n{prompt_text}{sep}{target_text}",
        assistant_prefix=f"<|sosp|>{audio_sp_token}",
    )


def build_edit_prompt(instruction: str, edit_type: str, audio_sp_token: str) -> str:
    """Speech editing.

    The semantic variant prepends "Identify the content of the audio." so the model
    emits <|sot|>{new text}<|eot|> before the audio, though it does not always do so.
    """
    if edit_type == "semantic":
        user_text = f"Identify the content of the audio. {instruction}"
    elif edit_type == "acoustic":
        user_text = instruction
    else:
        raise ValueError(f"unknown edit_type {edit_type!r}, expected semantic or acoustic")
    return _chatml(
        GENERIC_SYSTEM_PROMPT,
        f"Audio 1: <|sosp|>{audio_sp_token}<|eosp|>\n{user_text}",
    )


def build_voice_design_prompt(instruction: str, text: str) -> str:
    """Synthesis from a timbre description.

    The Chinese sentence between the two slots is the fixed training template and
    stays as-is even when instruction and text are English.
    """
    return _chatml(
        GENERIC_SYSTEM_PROMPT,
        f"{instruction}\n\n根据上述音色描述，合成以下文本对应的音频：\n{text}",
    )


# ============================================================== Output parsing
#
# The model returns its pieces concatenated; these helpers split them apart.
#
# Understanding with CoT yields "{reasoning}</think>\n\n{answer}". </think> is an
# ordinary vocabulary token that skip_special_tokens leaves in place, so a string
# split works. Without CoT it sits in the prompt rather than the output.
#
# Generation may embed <|sot|>{text}<|eot|> in text_ids: always for voice_design's
# timbre tags, and edit(semantic)'s rewritten text. Located by token id
# rather than by regex, since skip_special_tokens would eat the boundaries.

THINK_END = "</think>"


@dataclass
class UnderstandOutput:
    """Returned by asr / understand."""
    answer: str
    reasoning: str | None = None   # None without CoT, or when CoT was truncated


@dataclass
class AudioOutput:
    """Returned by tts / edit / voice_design."""
    audio: torch.Tensor            # (1, T) waveform at 24 kHz
    text: str | None               # text between <|sot|> and <|eot|>, if any
    vae_latents: torch.Tensor      # (1, N, 64) 25 Hz latents that `audio` decodes from
    text_ids: torch.Tensor         # (1, M) all text tokens, including <|sosp|>/<|eosp|>


def split_thinking(text: str) -> tuple[str | None, str]:
    """Split "{reasoning}</think>\\n\\n{answer}" into (reasoning, answer).

    Returns (None, text) unchanged when </think> is absent, which happens both
    without CoT and when CoT ran into max_new_tokens mid-reasoning.
    """
    if THINK_END not in text:
        return None, text
    reasoning, _, answer = text.partition(THINK_END)
    return reasoning.strip(), answer.strip()


def extract_sot_text(text_ids: torch.Tensor, tokenizer,
                     sot_id: int | None, eot_id: int | None) -> str | None:
    """Extract the text between <|sot|> and <|eot|> in text_ids, or None."""
    ids = text_ids[0].tolist() if text_ids.dim() == 2 else text_ids.tolist()
    if sot_id is None or sot_id not in ids:
        return None
    start = ids.index(sot_id)
    # Falls back to the end of the span when <|eot|> was never emitted
    end = ids.index(eot_id, start + 1) if eot_id in ids[start + 1:] else len(ids)
    span = ids[start + 1:end]
    return tokenizer.decode(span, skip_special_tokens=True) if span else None


# ================================================================ Inference

class FireRedAudioInference:
    """

    Args:
        model_path: Directory holding config.json and safetensors shards.
        tokenizer_path / processor_path: Default to model_path.
        vae_decoder_path: RedAE weights (.pt). Only generation tasks need it; without
            it those tasks raise when called.
        device: For example "cuda:0".
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        processor_path: str | None = None,
        vae_decoder_path: str | None = None,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        # generate needs left padding. Done here rather than inside AudioPromptEncoder
        # so every mutation to a shared object is visible in one place.
        self.tokenizer.padding_side = "left"
        self.processor = FireRedAudioProcessor.from_pretrained(processor_path or model_path)
        self.model = load_fireredaudio(model_path, device=self.device)

        self.encoder = AudioPromptEncoder(
            tokenizer=self.tokenizer,
            audio_processor=self.processor,
            audio_special_token=self.model.config.audio_special_token,
            audio_special_token_no_latent=self.model.config.audio_special_token_no_latent,
        )

        self._eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._pad_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        # Boundaries of the generation-side text output, used only when parsing results
        self._sot_id = self.tokenizer.convert_tokens_to_ids("<|sot|>")
        self._eot_id = self.tokenizer.convert_tokens_to_ids("<|eot|>")

        self.vae_decoder = None
        if vae_decoder_path is not None:
            self.vae_decoder = PretrainedRedAEAudioDecoderV1.from_pretrained(
                config=PretrainedRedAEDecoderConfig(), ckpt_path=vae_decoder_path
            ).to(self.device)

    # -------------------------------------------------------------- internals
    def _gen_config(self, task: str, max_new_tokens: int | None = None) -> GenerationConfig:
        params = dict(GENERATION_PARAMS[task])
        if max_new_tokens is not None:
            params["max_new_tokens"] = max_new_tokens
        return GenerationConfig(
            **params, eos_token_id=self._eos_id, pad_token_id=self._pad_id
        )

    def _read_generation_audio(self, path: str) -> torch.Tensor:
        """Generation-side reference audio, padded to a multiple of the patch rate."""
        return pad_to_multiple_of(read_audio(path, GENERATION_SAMPLE_RATE))

    def _run_audio_generation(self, task, input_chatml, input_audios,
                              **gen_kwargs) -> AudioOutput:
        """Shared path for generation tasks: encode -> generate_tts -> VAE decode."""
        if self.vae_decoder is None:
            raise ValueError(
                f"task={task!r} needs a VAE decoder to turn latents into a waveform; "
                "pass vae_decoder_path / --vae-decoder"
            )
        batch = self.encoder.encode(input_chatml, input_audios)
        text_ids, vae_latents = self.model.generate_tts(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            vae_audios=batch["vae_audios"].to(self.device),
            vae_is_assistant=batch["vae_is_assistant"].to(self.device),
            patch_encoder_output_attention_mask=(
                batch["patch_encoder_output_attention_mask"].to(self.device)
            ),
            generation_config=self._gen_config(task),
            **gen_kwargs,
        )
        if vae_latents.shape[1] == 0:
            # The model never entered audio mode. Passing this to the decoder would
            # fail deep inside Qwen3 with "cannot reshape tensor of 0 elements".
            raise RuntimeError(
                f"task={task!r} produced no audio (0 AE latents). The model stopped "
                f"after {text_ids.shape[1]} text tokens without emitting <|sosp|>. "
                "max_new_text_tokens is most likely too small: voice_design first "
                "writes a timbre description and edit(semantic) writes "
                "<|sot|>{new text}<|eot|>, either of which can run to hundreds of tokens."
            )
        audio = self.vae_decoder.decode(vae_latents.float())
        return AudioOutput(
            audio=audio,
            text=extract_sot_text(text_ids, self.tokenizer, self._sot_id, self._eot_id),
            vae_latents=vae_latents,
            text_ids=text_ids,
        )

    # ------------------------------------------------------ understanding tasks
    @torch.inference_mode()
    def understand(
        self,
        audio_paths: str | list[str],
        prompt: str,
        task: str = "understand",
        enable_thinking: bool = False,
        max_new_tokens: int | None = None,
    ) -> UnderstandOutput:
        """ASR and audio understanding.

        Args:
            audio_paths: One or more audios, rendered as "Audio 1: ... Audio 2: ..."
                in the prompt. Speaker verification passes two.
            prompt: The question.
            task: "asr" or "understand"; selects the sampling parameters.
            enable_thinking: Only valid for "understand"; see THINKING_TASKS.
            max_new_tokens: Defaults per task, or THINKING_MAX_NEW_TOKENS with CoT on.

        Returns:
            UnderstandOutput(answer, reasoning).
        """
        if enable_thinking and task not in THINKING_TASKS:
            raise ValueError(
                f"task={task!r} does not support enable_thinking: asr uses beam search "
                "for a deterministic transcript and its think block is always empty in "
                "training. Only task='understand' has CoT."
            )
        if max_new_tokens is None and enable_thinking:
            max_new_tokens = THINKING_MAX_NEW_TOKENS
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        input_chatml = build_understand_prompt(
            prompt, len(audio_paths), self.model.config.audio_special_token,
            enable_thinking=enable_thinking,
        )
        input_audios = [{
            "feat_type": FEAT_TYPE_UNDERSTAND,
            "audio_understand": read_audio(p, UNDERSTAND_SAMPLE_RATE).numpy(),
            "audio_generation": None,
            "role": "user",
        } for p in audio_paths]
        batch = self.encoder.encode(input_chatml, input_audios)
        out_ids = self.model.generate(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            audio_features=batch["audio_features"].to(self.device),
            audio_feature_attention_mask=batch["audio_feature_attention_mask"].to(self.device),
            generation_config=self._gen_config(task, max_new_tokens),
        )
        # generate runs on inputs_embeds and returns only the new tokens
        raw = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        reasoning, answer = split_thinking(raw)
        if enable_thinking and reasoning is None:
            logger.warning(
                "CoT is on but the output has no </think>: reasoning hit "
                "max_new_tokens=%s and was cut off, so what follows is partial "
                "reasoning rather than an answer. Raise --max-new-tokens.",
                max_new_tokens,
            )
        return UnderstandOutput(answer=answer, reasoning=reasoning)

    # --------------------------------------------------------- generation tasks
    @torch.inference_mode()
    def tts(
        self,
        prompt_text: str,
        prompt_audio: str,
        target_text: str,
        language: str = "zh",
        max_new_audio_steps: int = 750,
        min_new_audio_steps: int = 6,
        max_new_text_tokens: int = 512,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ) -> AudioOutput:
        """ICL voice cloning: the reference audio sits in the assistant turn and the
        model continues from its end.

        The prompt stops on <|sosp|> plus the placeholder, so generate_tts sees
        last_sosp > last_eosp and starts in audio mode.
        """
        input_chatml = build_tts_prompt(
            prompt_text, target_text, language,
            self.model.config.audio_special_token_no_latent,
        )
        input_audios = [{
            "feat_type": FEAT_TYPE_GENERATION,
            "audio_understand": None,
            "audio_generation": self._read_generation_audio(prompt_audio),
            "role": "assistant",
        }]
        return self._run_audio_generation(
            "tts", input_chatml, input_audios,
            max_new_audio_steps=max_new_audio_steps,
            min_new_audio_steps=min_new_audio_steps,
            max_new_text_tokens=max_new_text_tokens,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
        )

    @torch.inference_mode()
    def edit(
        self,
        audio_path: str,
        instruction: str,
        edit_type: str = "semantic",
        max_new_audio_steps: int = 750,
        min_new_audio_steps: int = 6,
        max_new_text_tokens: int = 512,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ) -> AudioOutput:
        """Speech editing.

        semantic changes content: the model writes <|sot|>{new text}<|eot|> before the
        audio, so max_new_text_tokens must be generous (a long Chinese sentence can
        exceed 100 tokens). acoustic changes pitch / rate / volume and
        needs only a single <|sosp|> in the text span.
        """
        input_chatml = build_edit_prompt(
            instruction, edit_type, self.model.config.audio_special_token_no_latent,
        )
        input_audios = [{
            "feat_type": FEAT_TYPE_GENERATION,
            "audio_understand": None,
            "audio_generation": self._read_generation_audio(audio_path),
            "role": "user",
        }]
        return self._run_audio_generation(
            "edit", input_chatml, input_audios,
            max_new_audio_steps=max_new_audio_steps,
            min_new_audio_steps=min_new_audio_steps,
            max_new_text_tokens=max_new_text_tokens,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
        )

    @torch.inference_mode()
    def voice_design(
        self,
        instruction: str,
        text: str,
        max_new_audio_steps: int = 750,
        min_new_audio_steps: int = 6,
        max_new_text_tokens: int = 512,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ) -> AudioOutput:
        """Synthesis from a timbre description.

        The model first writes a full structured description
        (<|sot|>[性别] 男 [年龄] 青年 [音高] 低 ...) and only then switches to audio,
        so max_new_text_tokens must be generous. generate_tts's default of 32 is far
        too small and yields zero latents.
        """
        return self._run_audio_generation(
            "voice_design",
            build_voice_design_prompt(instruction, text), [],
            max_new_audio_steps=max_new_audio_steps,
            min_new_audio_steps=min_new_audio_steps,
            max_new_text_tokens=max_new_text_tokens,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
        )


# ======================================================================= CLI

class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Keeps the module docstring's line breaks and still appends each default."""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=_HelpFormatter)
    p.add_argument("--task", required=True,
                   choices=list(UNDERSTAND_TASKS) + list(GENERATION_TASKS))
    p.add_argument("--model", required=True, help="model directory")
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--processor", default=None, help="defaults to --model")
    p.add_argument("--vae-decoder", default=None,
                   help="RedAE weights (.pt); required for tts / edit / voice_design")
    p.add_argument("--device", default="cuda:0", help="e.g. cuda:0, cuda:1, cpu")
    p.add_argument("--seed", type=int, default=None, help="left unseeded when omitted")
    p.add_argument("--output", default=None,
                   help="txt for understanding tasks (printed as well if omitted); "
                        "wav for generation tasks, which fall back to ./<task>.wav and "
                        "overwrite it on every run")

    p.add_argument("--audio", nargs="+", default=None,
                   help="input audio for asr / understand / edit; understand accepts "
                        "several (e.g. 2 for speaker verification), edit uses the first")
    p.add_argument("--prompt", default=None,
                   help=f"the question for understand; asr defaults to {DEFAULT_ASR_PROMPT!r}")
    p.add_argument("--prompt-audio", default=None, help="tts reference audio")
    p.add_argument("--prompt-text", default=None, help="transcript of the tts reference audio")
    p.add_argument("--target-text", default=None, help="text for tts to synthesize")
    p.add_argument("--language", default="zh", choices=["zh", "en"],
                   help="tts: en inserts a space between prompt and target text")
    p.add_argument("--instruction", default=None,
                   help="edit instruction, or voice_design timbre description")
    p.add_argument("--edit-type", default="semantic", choices=["semantic", "acoustic"],
                   help="semantic rewrites the content, acoustic changes pitch / rate / "
                        "volume / denoising")
    p.add_argument("--enable-thinking", action="store_true",
                   help="leave the think block open at <think>\\n so the model writes its "
                        "own CoT. understand only: asr uses beam search, and the "
                        "generation tasks train with an empty think block")
    p.add_argument("--text", default=None, help="text for voice_design to synthesize")

    p.add_argument("--max-new-audio-steps", type=int, default=750,
                   help="~160 ms per step")
    p.add_argument("--min-new-audio-steps", type=int, default=6,
                   help="suppress <|eosp|> for the first N audio steps, so generation "
                        "cannot stop immediately")
    p.add_argument("--max-new-text-tokens", type=int, default=512,
                   help="cap on the text tokens emitted before switching to audio; "
                        "edit(semantic) writes the rewritten text and voice_design a "
                        "timbre description first, and too small a value yields zero latents")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help=f"text length cap for asr / understand, default 300; raised to "
                        f"{THINKING_MAX_NEW_TOKENS} when understand runs with "
                        "--enable-thinking. Generation tasks use --max-new-text-tokens")
    p.add_argument("--n-timesteps", type=int, default=10, help="flow matching denoising steps")
    p.add_argument("--inference-cfg", type=float, default=2.0,
                   help="classifier-free guidance weight w, applied to the velocity "
                        "field as (1+w)*cond - w*uncond; w=0 disables guidance and "
                        "larger w pushes further toward the conditional")
    return p.parse_args()


def _require(args, *names):
    missing = [f"--{n.replace('_', '-')}" for n in names if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"--task {args.task} requires: {', '.join(missing)}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    if args.enable_thinking and args.task not in THINKING_TASKS:
        raise SystemExit(
            f"--task {args.task} does not support --enable-thinking: "
            + ("asr uses beam search for a deterministic transcript and trains with "
               "an empty think block. " if args.task == "asr" else
               "generation tasks train with an empty think block and wrap their text "
               "output in <|sot|>…<|eot|>. ")
            + f"Only --task {'/'.join(THINKING_TASKS)} supports it.")
    if args.task in GENERATION_TASKS:
        _require(args, "vae_decoder")
    engine = FireRedAudioInference(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        processor_path=args.processor,
        vae_decoder_path=args.vae_decoder,
        device=args.device,
    )

    if args.task in UNDERSTAND_TASKS:
        _require(args, "audio")
        prompt = args.prompt or (DEFAULT_ASR_PROMPT if args.task == "asr" else None)
        if prompt is None:
            raise SystemExit("--task understand requires: --prompt")
        result = engine.understand(args.audio, prompt, task=args.task,
                                   enable_thinking=args.enable_thinking,
                                   max_new_tokens=args.max_new_tokens)
        if result.reasoning is not None:
            print("===== reasoning =====")
            print(result.reasoning)
            print("===== answer =====")
        print(result.answer)
        if args.output:
            # Answer only; this file usually feeds WER or other scoring
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.answer + "\n")
            logger.info("wrote text to %s", args.output)
        return

    common = dict(
        max_new_audio_steps=args.max_new_audio_steps,
        min_new_audio_steps=args.min_new_audio_steps,
        max_new_text_tokens=args.max_new_text_tokens,
        n_timesteps=args.n_timesteps,
        inference_cfg=args.inference_cfg,
    )
    if args.task == "tts":
        _require(args, "prompt_audio", "prompt_text", "target_text")
        result = engine.tts(
            args.prompt_text, args.prompt_audio, args.target_text, args.language, **common,
        )
    elif args.task == "edit":
        _require(args, "audio", "instruction")
        result = engine.edit(args.audio[0], args.instruction, args.edit_type, **common)
    else:
        _require(args, "instruction", "text")
        result = engine.voice_design(args.instruction, args.text, **common)

    if result.text is not None:
        # Rewritten text from edit(semantic), or voice_design's timbre tags
        print("===== text output =====")
        print(result.text)
    out = args.output or f"{args.task}.wav"
    torchaudio.save(out, result.audio.cpu().float(), sample_rate=GENERATION_SAMPLE_RATE)
    logger.info("wrote audio to %s (%.2fs)", out,
                result.audio.shape[-1] / GENERATION_SAMPLE_RATE)


if __name__ == "__main__":
    main()
