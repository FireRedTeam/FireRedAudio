import warnings

import torch
from transformers import GenerationConfig, PreTrainedModel
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack

from .configuration_fireredaudio import FireRedAudioConfig
from .audio_encoder.modeling_audio_encoder import FireRedAudioEncoder
from .redae.encoder import RedAEAudioEncoderV1
from .flow.estimator import RedDiT
from .flow.patch_encoder import RedPatchEncoder


class FireRedAudioForCausalLM(PreTrainedModel):
    config_class = FireRedAudioConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: FireRedAudioConfig):
        super().__init__(config)

        # backbone llm
        self.backbone_llm = Qwen3_5ForCausalLM(config=self.config.backbone_config)

        # audio encoder
        self.audio_encoder = FireRedAudioEncoder(self.config.audio_encoder_config)

        # red_vae
        self.red_vae = RedAEAudioEncoderV1(self.config.red_vae_config)
        self.red_vae.eval()

        # patch encoder
        self.patch_encoder = RedPatchEncoder(self.config.patch_encoder_config)

        # dit head
        self.dit = RedDiT(self.config.dit_config)

        self.post_init()

    def train(self, mode: bool = True):
        super().train(mode)
        self.red_vae.eval()
        return self

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = None

        audio_feat_lengths, audio_output_lengths = (
            self.audio_encoder._get_feat_extract_output_lengths(
                audio_feature_lengths
                if audio_feature_lengths is not None
                else feature_attention_mask.sum(-1)
            )
        )
        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )
        audio_outputs = self.audio_encoder(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
            return_dict=True,
            **kwargs,
        )
        if audio_outputs.shape[0] != sum(audio_output_lengths.tolist()):
            raise ValueError(
                "length of audio_features should match audio_output_lengths"
            )

        return audio_outputs

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        audio_features=None,
        audio_feature_attention_mask=None,
        **kwargs,
    ):
        # Step1. token embeddings
        input_embeds = self.backbone_llm.model.get_input_embeddings()(input_ids)

        # Step2. audio encoder features
        audio_encoder_output = self.get_audio_features(
            input_features=audio_features,
            feature_attention_mask=audio_feature_attention_mask,
        )

        audio_placeholder_mask_2d = input_ids == self.config.audio_special_token_id
        num_placeholders = audio_placeholder_mask_2d.sum().item()
        if audio_encoder_output.shape[0] != num_placeholders:
            raise ValueError(
                f"audio_encoder_output length ({audio_encoder_output.shape[0]}) "
                f"!= num_placeholders ({num_placeholders})"
            )
        audio_placeholder_mask_2d = audio_placeholder_mask_2d.unsqueeze(-1).expand_as(input_embeds)
        input_embeds = input_embeds.masked_scatter(audio_placeholder_mask_2d, audio_encoder_output)

        # Step3. delegate to backbone LLM generate
        return self.backbone_llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
    
    # --- Flow one-step generation
    def _flow_onestep_generate(
        self, 
        backbone_output: torch.Tensor,
        history_vae_latents: torch.Tensor,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ):
        """

        Args:
            backbone_output(torch.Tensor): full length backbone output, shape (b=1, t', c)
            history_vae_latents(torch.Tensor): history generated vae latents, shape (b=1, t'', c)
            n_timesteps(int): default to 10
        Return:
            one_vae_latents(torch.Tensor): this step generated vae latents
            next_backbone_input_embeds(torch.Tensor): next backbone step patched latents
        """
        one_vae_latents = self.dit.generate(
            backbone_output=backbone_output,
            history_vae_latents=history_vae_latents,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
        )
        next_backbone_input_embeds = self.patch_encoder(one_vae_latents)
        return one_vae_latents, next_backbone_input_embeds        

    @torch.inference_mode()
    def generate_tts(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        vae_audios: torch.Tensor | None = None,
        vae_is_assistant: torch.Tensor | None = None,
        patch_encoder_output_attention_mask: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        max_new_audio_steps: int = 750,
        min_new_audio_steps: int = 0,
        max_new_text_tokens: int = 32,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ):
        """Hybrid AR generation: text token sampling plus audio chunk DiT denoising.

        Text steps sample from lm_head through HF LogitsProcessor; audio steps call
        _flow_onestep_generate for 4 consecutive AE latents per chunk. <|sosp|> and
        <|eosp|> switch between the two modes. batch_size=1 only.

        Audio on both the user and assistant side goes through the generation path
        (VAE + patch_encoder), so audio_features / audio_feature_attention_mask are
        not exposed here.

        Args:
            input_ids: (1, T_p) chatml token ids with placeholders already expanded.
            attention_mask: (1, T_p) left padded, same shape as input_ids.
            vae_audios / vae_is_assistant / patch_encoder_output_attention_mask:
                Optional generation-side reference audio. vae_is_assistant marks which
                of them sit in the assistant turn; generation continues from the last
                such one, which is how ICL voice cloning works. Audio given in the user
                turn leaves it False and serves as context only.
            generation_config: Controls text sampling (do_sample / temperature / top_p /
                top_k / repetition_penalty / eos_token_id).
            max_new_audio_steps: Cap on audio chunks; each step is 4 AE latents (~160 ms).
            min_new_audio_steps: Suppress <|eosp|> for the first N steps so generation
                cannot stop immediately. N=6 is ~0.96 s; 0 disables it.
            max_new_text_tokens: Cap on text steps, guarding against a text-mode loop.
            n_timesteps / inference_cfg: Passed through to _flow_onestep_generate.

        Returns:
            text_token_ids: (1, T_gen) newly generated text tokens, including
                sosp / eosp / im_end.
            vae_latents: (1, N*4, 64) AE latents in generation order, ready for an
                external VAE decoder.
        """
        # ---- Phase 0: argument resolution ----
        if input_ids.shape[0] != 1:
            raise NotImplementedError("generate_tts only supports batch_size=1 for now")

        gen_cfg = generation_config if generation_config is not None else GenerationConfig()

        raw = gen_cfg.eos_token_id
        if raw is None:
            raise ValueError("generation_config.eos_token_id is required")
        eos_set = {int(x) for x in raw} if isinstance(raw, (list, tuple)) else {int(raw)}

        sosp_id = self.config.sosp_idx
        eosp_id = self.config.eosp_idx
        audio_no_latent_id = self.config.audio_special_no_latent_id
        device = input_ids.device

        # Text-side LogitsProcessor (repetition_penalty / temperature / top_k / top_p)
        from transformers.generation.logits_process import (
            RepetitionPenaltyLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )
        text_logits_processor = LogitsProcessorList()
        if gen_cfg.repetition_penalty is not None and float(gen_cfg.repetition_penalty) != 1.0:
            text_logits_processor.append(
                RepetitionPenaltyLogitsProcessor(penalty=float(gen_cfg.repetition_penalty))
            )
        if gen_cfg.do_sample:
            if gen_cfg.temperature is not None and float(gen_cfg.temperature) != 1.0:
                text_logits_processor.append(
                    TemperatureLogitsWarper(temperature=float(gen_cfg.temperature))
                )
            if gen_cfg.top_k is not None and int(gen_cfg.top_k) > 0:
                text_logits_processor.append(TopKLogitsWarper(top_k=int(gen_cfg.top_k)))
            if gen_cfg.top_p is not None and float(gen_cfg.top_p) < 1.0:
                text_logits_processor.append(TopPLogitsWarper(top_p=float(gen_cfg.top_p)))

        # ---- Phase 1: Prefill ----
        # Step1. token embeddings
        input_embeds = self.backbone_llm.model.get_input_embeddings()(input_ids)

        # Step2. Scatter generation-side ref audio (VAE + patch_encoder) into <|AUDIO_NO_LATENT|>
        ref_vae_latents = None
        if vae_audios is not None and vae_audios.shape[0] > 0:
            ref_vae_latents = self.red_vae.encode(vae_audios).transpose(1, 2)  # (N_gen, T_vae_max, 64)
            patch_enc_latent = self.patch_encoder(ref_vae_latents, None)        # (N_gen, T_vae_max//4, H)
            patch_enc_raged = patch_enc_latent[patch_encoder_output_attention_mask]
            mask_gen = (input_ids == audio_no_latent_id)
            num_gen_ph = int(mask_gen.sum().item())
            if patch_enc_raged.shape[0] != num_gen_ph:
                raise ValueError(
                    f"patch_encoder_output length ({patch_enc_raged.shape[0]}) "
                    f"!= num_placeholders ({num_gen_ph})"
                )
            input_embeds = input_embeds.masked_scatter(
                mask_gen.unsqueeze(-1).expand_as(input_embeds),
                patch_enc_raged,
            )

        # Prefill backbone forward, building the KV cache
        out = self.backbone_llm.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_kv = out.past_key_values
        prefill_h = out.last_hidden_state           # (1, T_p, H), kept whole for Phase 1.5
        current_h = prefill_h[:, -1:]               # (1, 1, H)
        attn_mask = attention_mask                  # one column appended per step

        embed_tokens = self.backbone_llm.model.get_input_embeddings()

        # Pick the initial mode and preload backbone_audio_hiddens
        ids_1d = input_ids[0]
        sosp_positions = (ids_1d == sosp_id).nonzero(as_tuple=True)[0]
        eosp_positions = (ids_1d == eosp_id).nonzero(as_tuple=True)[0]
        last_sosp_pos = int(sosp_positions[-1].item()) if sosp_positions.numel() > 0 else -1
        last_eosp_pos = int(eosp_positions[-1].item()) if eosp_positions.numel() > 0 else -1
        initial_mode = "audio" if last_sosp_pos > last_eosp_pos else "text"

        H = current_h.shape[-1]
        vae_channels = self.dit.config.vae_channels
        empty_h = torch.empty((1, 0, H), dtype=current_h.dtype, device=device)

        if initial_mode == "audio":
            backbone_audio_hiddens = prefill_h[:, last_sosp_pos:-1].contiguous()  # (1, K-1, H)
        else:
            backbone_audio_hiddens = empty_h  # (1, 0, H)

        # Release the full prefill_h
        prefill_h = None

        # ---- Phase 2: Hybrid AR loop ----
        mode = initial_mode
        generated_text_ids: list[int] = []
        generated_vae_latents = torch.empty(
            (1, 0, vae_channels), dtype=current_h.dtype, device=device
        )

        if (
            initial_mode == "audio"
            and ref_vae_latents is not None
            and patch_encoder_output_attention_mask is not None
            and vae_is_assistant is not None
            and bool(vae_is_assistant.any())
        ):
            assistant_idx = vae_is_assistant.nonzero(as_tuple=True)[0]
            last_idx = int(assistant_idx[-1].item())
            valid_patch_len = int(patch_encoder_output_attention_mask[last_idx].sum().item())
            valid_vae_len = valid_patch_len * self.dit.patch_size
            if valid_vae_len > 0:
                history_vae_latents = ref_vae_latents[last_idx : last_idx + 1, :valid_vae_len].contiguous()
            else:
                history_vae_latents = None
        else:
            history_vae_latents = None

        n_text_tokens_emitted = 0
        n_audio_steps_emitted = 0

        while True:
            if mode == "text":
                # Sample a text token
                logits = self.backbone_llm.lm_head(current_h[:, -1, :])  # (1, V)
                if generated_text_ids:
                    input_ids_so_far = torch.tensor(
                        [generated_text_ids], dtype=torch.long, device=device
                    )
                else:
                    input_ids_so_far = torch.empty((1, 0), dtype=torch.long, device=device)
                scores = text_logits_processor(input_ids_so_far, logits)
                if gen_cfg.do_sample:
                    probs = torch.softmax(scores.float(), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)[:, 0]  # (1,)
                else:
                    next_token = scores.argmax(dim=-1)                          # (1,)

                tok = int(next_token.item())
                generated_text_ids.append(tok)
                n_text_tokens_emitted += 1

                # Stop
                if tok in eos_set:
                    break

                # Mode switch
                next_mode = "audio" if tok == sosp_id else "text"

                # Feed next_token's embedding to the backbone, mode switch or not
                next_embed = embed_tokens(next_token.view(1, 1))  # (1, 1, H)
                attn_mask = torch.cat(
                    [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=device)],
                    dim=1,
                )
                out = self.backbone_llm.model(
                    inputs_embeds=next_embed,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = out.past_key_values
                current_h = out.last_hidden_state  # (1, 1, H)
                mode = next_mode

                # Guard against a text-mode loop
                if mode == "text" and n_text_tokens_emitted >= max_new_text_tokens:
                    warnings.warn(
                        f"generate_tts: text mode hit max_new_text_tokens={max_new_text_tokens} "
                        "without producing <|sosp|>; returning early."
                    )
                    break

            else:  # mode == "audio"
                # Collect backbone hiddens for the audio span; DiT.generate conditions
                # on only the last 3 frames.
                backbone_audio_hiddens = torch.cat(
                    [backbone_audio_hiddens, current_h[:, -1:]], dim=1
                )  # (1, K+1, H)

                # One patch = 4 AE latents
                one_vae_latents, next_input_embeds = self._flow_onestep_generate(
                    backbone_output=backbone_audio_hiddens,
                    history_vae_latents=history_vae_latents,
                    n_timesteps=n_timesteps,
                    inference_cfg=inference_cfg,
                )
                # one_vae_latents: (1, 4, 64); next_input_embeds: (1, 1, H)

                generated_vae_latents = torch.cat(
                    [generated_vae_latents, one_vae_latents], dim=1
                )
                history_vae_latents = (
                    one_vae_latents
                    if history_vae_latents is None
                    else torch.cat([history_vae_latents, one_vae_latents], dim=1)
                )
                n_audio_steps_emitted += 1

                # Feed patch_encoder output back in as continuous embeddings, not token ids
                attn_mask = torch.cat(
                    [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=device)],
                    dim=1,
                )
                out = self.backbone_llm.model(
                    inputs_embeds=next_input_embeds,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = out.past_key_values
                current_h = out.last_hidden_state  # (1, 1, H)

                # <|eosp|> is the only exit signal. The other placeholder positions do
                # carry a next-token loss, but at ce_weight 0.01 against 1.0 elsewhere,
                # so lm_head is only weakly supervised there and its argmax means little.
                audio_exit_logits = self.backbone_llm.lm_head(current_h[:, -1, :])  # (1, V)
                if n_audio_steps_emitted < min_new_audio_steps:
                    audio_exit_logits[:, eosp_id] = float("-inf")
                next_argmax = int(audio_exit_logits.argmax(dim=-1).item())

                if next_argmax == eosp_id:
                    generated_text_ids.append(eosp_id)
                    eosp_embed = embed_tokens(
                        torch.tensor([[eosp_id]], dtype=torch.long, device=device)
                    )
                    attn_mask = torch.cat(
                        [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=device)],
                        dim=1,
                    )
                    out = self.backbone_llm.model(
                        inputs_embeds=eosp_embed,
                        attention_mask=attn_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_kv = out.past_key_values
                    current_h = out.last_hidden_state
                    mode = "text"
                    backbone_audio_hiddens = empty_h  # reset to (1, 0, H)
                elif n_audio_steps_emitted >= max_new_audio_steps:
                    warnings.warn(
                        f"generate_tts: hit max_new_audio_steps={max_new_audio_steps} "
                        "without producing <|eosp|>; returning truncated latents."
                    )
                    break

        # ---- Phase 3: Return ----
        if generated_text_ids:
            text_token_ids = torch.tensor([generated_text_ids], dtype=torch.long, device=device)
        else:
            text_token_ids = torch.empty((1, 0), dtype=torch.long, device=device)

        return text_token_ids, generated_vae_latents  # (1, N*4, 64) or (1, 0, 64)



from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("firered_audio", FireRedAudioConfig)
AutoModelForCausalLM.register(FireRedAudioConfig, FireRedAudioForCausalLM)
