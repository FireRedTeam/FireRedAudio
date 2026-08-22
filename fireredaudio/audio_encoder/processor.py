from transformers.processing_utils import ProcessorMixin

from ..utils.audio import UNDERSTAND_SAMPLE_RATE


class FireRedAudioProcessor(ProcessorMixin):
    attributes = ["feature_extractor"]

    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)

    def __call__(self, audios, **kwargs):
        kwargs["padding"] = "max_length"  # Support "max_length" padding only here
        kwargs['sampling_rate'] = UNDERSTAND_SAMPLE_RATE
        kwargs['return_attention_mask'] = True
        kwargs['return_tensors'] = 'pt'

        audio_inputs = self.feature_extractor(audios, **kwargs)
        audio_inputs["feature_attention_mask"] = audio_inputs.pop(
            "attention_mask"
        )
        audio_inputs["input_features"] = audio_inputs.pop(
            "input_features"
        ).bfloat16()

        return audio_inputs
