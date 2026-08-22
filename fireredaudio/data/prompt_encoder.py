"""Encode a chatml string and its audios into model inputs.

Each audio appears as a single placeholder token in the prompt and is expanded here
to exactly the encoder output length, as `masked_scatter` in the model requires:

    <|AUDIO|>            12.5 Hz
    <|AUDIO_NO_LATENT|>  25 Hz latents grouped by patch_size=4, i.e. 6.25 Hz
"""

import torch

from ..audio_encoder.modeling_audio_encoder import FireRedAudioEncoder
from ..utils.audio import UNDERSTAND_SAMPLE_RATE
from ..redae.encoder import get_vae_and_patch_output_len

FEAT_TYPE_UNDERSTAND = 'feat_understand'
FEAT_TYPE_GENERATION = 'feat_generation'


class AudioPromptEncoder:

    def __init__(
        self,
        tokenizer,
        audio_processor,
        audio_special_token,
        audio_special_token_no_latent,
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.audio_special_token = audio_special_token
        self.audio_special_token_no_latent = audio_special_token_no_latent

        # Expansion replaces placeholders one at a time on the string, so neither
        # token may be a substring of the other.
        assert self.audio_special_token not in self.audio_special_token_no_latent and self.audio_special_token_no_latent not in self.audio_special_token

    def _extract_audio_feature(self, audio_arrays):
        # Empty for generation tasks (tts / edit / voice_design)
        if not audio_arrays:
            return {
                "audio_features": torch.empty(0, 128, 1, dtype=torch.bfloat16),
                "audio_feature_attention_mask": torch.empty(
                    0, 1, dtype=torch.int32
                ),
            }

        # Longest audio + 1 s, keeping audio_len < pad_len to avoid STFT reflect
        # boundary effects.
        max_len_samples = max(len(a) for a in audio_arrays) + UNDERSTAND_SAMPLE_RATE
        res = self.audio_processor(audios=audio_arrays, max_length=max_len_samples)
        res["audio_features"] = res.pop("input_features")
        res["audio_feature_attention_mask"] = res.pop("feature_attention_mask")
        return res

    def pad_and_mask(self, tensor_list, padding_value=0):
        # Empty for understanding tasks (asr / understand)
        if not tensor_list:
            return torch.empty(0, 0), torch.empty(0, 0)

        padded = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)

        lengths = torch.tensor([get_vae_and_patch_output_len(t)[1] for t in tensor_list])
        max_len = max(lengths)

        attention_mask = (torch.arange(max_len)[None, :] < lengths[:, None])

        return padded.to(torch.bfloat16), attention_mask

    def encode(self, chatml: str, audios: list[dict]) -> dict:
        """
        Args:
            chatml: Prompt string, one placeholder token per audio.
            audios: In the same order as the placeholders, each entry shaped like
                {"feat_type": FEAT_TYPE_UNDERSTAND, "audio_understand": <16 kHz np.ndarray>,
                 "audio_generation": None, "role": "user"}

        Returns:
            input_ids, attention_mask, audio_features, audio_feature_attention_mask,
            vae_audios, patch_encoder_output_attention_mask, vae_is_assistant
        """
        # for audio encoder
        audio_arrays = []
        # for vae
        vae_audio_arrays = []
        vae_is_assistant = []
        for audio in audios:
            if audio['feat_type'] == FEAT_TYPE_UNDERSTAND:
                audio_arrays.append(audio['audio_understand'])
            elif audio['feat_type'] == FEAT_TYPE_GENERATION:
                vae_audio_arrays.append(audio['audio_generation'])
                vae_is_assistant.append(audio['role'] == 'assistant')

        res = self._extract_audio_feature(audio_arrays)
        _, replace_str_lengths = FireRedAudioEncoder._get_feat_extract_output_lengths(res['audio_feature_attention_mask'].sum(-1))
        replace_str_lengths = replace_str_lengths.tolist()

        num_audio_tokens = (chatml.count(self.audio_special_token)
                            + chatml.count(self.audio_special_token_no_latent))
        num_audios = len(audio_arrays) + len(vae_audio_arrays)
        if num_audio_tokens != num_audios:
            raise ValueError(
                f"num_audio_tokens != num_audios: {num_audio_tokens} != {num_audios}"
            )

        # The two placeholder kinds must be replaced in audio order rather than all
        # at once, so substitute a sentinel first and fill it back afterwards.
        place_holder = '__FIREREDAUDIO__<placeholder>__FIREREDAUDIO__'
        replace_str = []
        s = chatml
        for audio in audios:
            if audio['feat_type'] == FEAT_TYPE_UNDERSTAND:
                s = s.replace(self.audio_special_token, place_holder, 1)
                expanded_audio_special_token = self.audio_special_token * replace_str_lengths.pop(0)
            elif audio['feat_type'] == FEAT_TYPE_GENERATION:
                s = s.replace(self.audio_special_token_no_latent, place_holder, 1)
                expanded_audio_special_token = self.audio_special_token_no_latent * get_vae_and_patch_output_len(audio['audio_generation'])[1]
            replace_str.append(expanded_audio_special_token)

        while place_holder in s:
            s = s.replace(place_holder, replace_str.pop(0), 1)

        input_ids = self.tokenizer(
            [s],
            return_tensors="pt",
            truncation=False,
            padding=True,
            add_special_tokens=False,
        )
        res["attention_mask"] = input_ids['attention_mask']
        res["input_ids"] = input_ids['input_ids']

        # gen_audios
        vae_audios, gen_audios_vae_attention_mask = self.pad_and_mask(vae_audio_arrays)
        res['vae_audios'] = vae_audios
        res['patch_encoder_output_attention_mask'] = gen_audios_vae_attention_mask
        res['vae_is_assistant'] = torch.tensor(vae_is_assistant, dtype=torch.bool)

        return res
