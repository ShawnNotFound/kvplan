"""vLLM 0.32's multimodal startup profiling: which dummy items it encodes and what that costs.

Before the text forward pass, GPUModelRunner.profile_run (vllm/v1/worker/gpu/model_runner.py)
asks MultiModalBudget (vllm/multimodal/encoder_budget.py) for dummy encoder inputs, runs the
encoder on them in one batch (EncoderRunner.profile_encoder_cache,
vllm/v1/worker/gpu/mm/encoder_runner.py), keeps the outputs in the encoder cache, and then runs
the text model. The profiled peak is the larger of the encoder pass and the text pass with the
encoder outputs still held.

1. Each enabled modality gets a maximum number of tokens per item from the model's processing
   info (get_mm_max_tokens_per_item); a modality is enabled unless --limit-mm-per-prompt sets it to 0.
2. The modality with the most tokens is profiled; ties go to the later name ("video" > "image").
3. Encoder budget = max(max_num_batched_tokens, largest item) (compute_mm_encoder_budget,
   vllm/v1/core/encoder_cache_manager.py); the batch holds
   max(1, min(budget // item, max_num_seqs * max(1, min(limit, max_model_len // item)))) items
   (MultiModalBudget._get_max_items).
4. The items themselves come from the model's dummy-input builder and its HF processor, which can
   produce fewer tokens than step 1 claims (Qwen2.5-Omni's video processor caps frames at
   768 x 28 x 28 pixels).

Only the Qwen2-VL family (Qwen2-VL, Qwen2.5-VL, Qwen2.5-Omni) is derived exactly; other models get
an assumed item of max_num_batched_tokens tokens and are labelled approximate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MAX_FRAMES_PER_VIDEO = 14            # vllm/model_executor/models/qwen2_vl.py _MAX_FRAMES_PER_VIDEO
OMNI_VIDEO_MAX_PIXELS = 768 * 28 * 28  # transformers Qwen2_5OmniProcessor videos_kwargs size.longest_edge
OMNI_VIDEO_MIN_PIXELS = 128 * 28 * 28
QWEN_VL_TYPES = {"qwen2_vl", "qwen2_5_vl"}
QWEN_OMNI_TYPES = {"qwen2_5_omni"}


@dataclass
class MMProfile:
    modality: str
    items: int
    item_tokens: int           # tokens per item vLLM budgets for (step 1)
    item_outputs: int          # tokens per item the dummy input actually produces (step 4)
    item_positions: int        # encoder positions per item (before spatial merging)
    budget: int
    encoder_peak_bytes: int    # peak of the encoder pass above the starting allocation
    output_bytes: int          # encoder outputs held while the text pass runs
    approximate: bool
    notes: list[str] = field(default_factory=list)


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """transformers/models/qwen2_vl/image_processing_qwen2_vl.py smart_resize."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_size_with_most_features(max_pixels: int, unit: int) -> tuple[int, int]:
    """(width, height) of vLLM's largest dummy image (Qwen2VLProcessingInfo.get_image_size_with_most_features)."""
    max_seq_len = max_pixels // (unit * unit)

    def closest_factor_pair(n: int) -> tuple[int, int]:
        for d in range(math.isqrt(n), 0, -1):
            if n % d == 0:
                return d, n // d
        return 1, n

    height_factor, width_factor = 1, max_seq_len
    for seq_len in range(max_seq_len, 0, -1):
        height_factor, width_factor = closest_factor_pair(seq_len)
        if width_factor / height_factor <= 200:
            break
    return unit * width_factor, unit * height_factor


def _qwen_tokens(height: int, width: int, frames: int, patch: int, merge: int, temporal: int,
                 min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """(tokens, encoder positions) of one image or video after smart_resize (_get_vision_info)."""
    h, w = smart_resize(height, width, patch * merge, min_pixels, max_pixels)
    grid_t = max((frames + (-frames % temporal)) // temporal, 1)
    positions = grid_t * (h // patch) * (w // patch)
    return positions // merge ** 2, positions


def _items_per_batch(budget: int, item: int, num_seqs: int, max_model_len: int, limit: int | None) -> int:
    per_prompt = max(1, min(limit if limit is not None else 10**9, max_model_len // item))
    return max(1, min(budget // item, num_seqs * per_prompt))


def qwen_vl_profile(model_type: str, vision: dict, preprocessor: dict | None, batched: int, num_seqs: int,
                    max_model_len: int, limits: dict[str, int], max_pixels: int | None,
                    min_pixels: int | None) -> MMProfile | None:
    """The dummy batch vLLM 0.32 encodes at startup for the Qwen2-VL family."""
    patch = vision.get("patch_size", 14)
    merge = vision.get("spatial_merge_size", 2)
    temporal = vision.get("temporal_patch_size", 2)
    unit = patch * merge
    pre = preprocessor or {}
    size = pre.get("size") if isinstance(pre.get("size"), dict) else {}
    img_max = max_pixels or pre.get("max_pixels") or size.get("longest_edge")
    img_min = min_pixels or pre.get("min_pixels") or size.get("shortest_edge") or 56 * 56
    if not img_max:
        return None
    width, height = image_size_with_most_features(img_max, unit)
    image_tokens, image_positions = _qwen_tokens(height, width, 1, patch, merge, temporal, img_min, img_max)

    # get_num_frames_with_most_features: as many frames as fit in max_model_len tokens, at most 14.
    frames = 1
    while _qwen_tokens(height, width, frames + 1, patch, merge, temporal, img_min, img_max)[0] <= max_model_len:
        frames += 1
    frames = max(min(frames, MAX_FRAMES_PER_VIDEO), 1)
    video_tokens, video_positions = _qwen_tokens(height, width, frames, patch, merge, temporal, img_min, img_max)
    video_outputs = video_tokens
    if model_type in QWEN_OMNI_TYPES:
        # The Omni HF processor resizes video frames with its own size unless max_pixels is given.
        v_max = max_pixels or OMNI_VIDEO_MAX_PIXELS
        v_min = min_pixels or OMNI_VIDEO_MIN_PIXELS
        video_outputs, video_positions = _qwen_tokens(height, width, frames, patch, merge, temporal, v_min, v_max)

    candidates = {"image": (image_tokens, image_tokens, image_positions),
                  "video": (video_tokens, video_outputs, video_positions)}
    enabled = {m: v for m, v in candidates.items() if limits.get(m, 1) != 0}
    if not enabled:
        return None
    all_tokens = [v[0] for v in enabled.values()]
    if model_type in QWEN_OMNI_TYPES and limits.get("audio", 1) != 0:
        all_tokens.append(750)   # 30 s of audio: 3000 mel frames -> 1500 encoder positions -> 750 tokens
    modality = max(enabled, key=lambda m: (enabled[m][0], m))
    tokens, outputs, positions = enabled[modality]
    budget = max(batched, max(all_tokens))
    items = _items_per_batch(budget, tokens, num_seqs, max_model_len, limits.get(modality))
    return MMProfile(modality=modality, items=items, item_tokens=tokens, item_outputs=outputs,
                     item_positions=positions, budget=budget, encoder_peak_bytes=0, output_bytes=0,
                     approximate=False)


# ------------------------------------------------------------------- encoder memory per position
#
# Live tensors at the encoder's peak, per encoder position, in the model dtype (2 bytes), read from
# the encoder code that vLLM 0.32 runs in eager mode:
#
# qwen_vit (Qwen2_5_VisionBlock, vllm/model_executor/models/qwen2_5_vl.py): inside the MLP the
#   block holds the fused-norm input and the residual (2h), gate_up (2i/tp), the activation output
#   (i/tp) and down_proj's output (h); with tp > 1 the all-reduce returns one more h-sized tensor.
#   The pixel input (3 x 2 x 14 x 14 values per position) stays allocated for the whole pass:
#   uint8 for Qwen2.5-VL, whose vLLM processor normalises on the GPU (visual.input_norm), and
#   bf16 for Qwen2.5-Omni, whose HF processor normalises.
# whisper (transformers' Qwen2AudioEncoderLayer, run unsplit on every GPU): residual and normed
#   input (2h) plus fc1's output and the GELU output (2i). This one is approximate: it was 25% below
#   the one measured encoder peak (Qwen2-Audio), where the text pass set the peak anyway.
# generic: the qwen_vit rule applied to the encoder config; approximate.

DTYPE_BYTES = 2
QWEN_VIT_PIXELS_PER_POSITION = 3 * 2 * 14 * 14


def bytes_per_position(enc, tp: int) -> int:
    h = enc.hidden_size or 0
    i = enc.intermediate_size or 4 * h
    if enc.family == "whisper":
        return (2 * h + 2 * i) * DTYPE_BYTES + 2 * 128 * (4 + DTYPE_BYTES)   # + 2 mel frames, fp32 and bf16
    per = (3 * h + 3 * i // tp + (h if tp > 1 else 0)) * DTYPE_BYTES
    if enc.family == "qwen_vit":
        per += QWEN_VIT_PIXELS_PER_POSITION * (1 if enc.model_type == "qwen2_5_vl" else DTYPE_BYTES)
    return per


def audio_profile(batched: int, num_seqs: int, max_model_len: int, limits: dict[str, int]) -> MMProfile | None:
    """Whisper-style audio: 30 s per item, 3000 mel frames -> 1500 encoder positions -> 750 tokens
    (Qwen2-Audio and Qwen2.5-Omni processors)."""
    if limits.get("audio", 1) == 0:
        return None
    tokens = 750
    budget = max(batched, tokens)
    items = _items_per_batch(budget, tokens, num_seqs, max_model_len, limits.get("audio"))
    return MMProfile(modality="audio", items=items, item_tokens=tokens, item_outputs=tokens, item_positions=1500,
                     budget=budget, encoder_peak_bytes=0, output_bytes=0, approximate=False)


def profile(enc, text_hidden: int, tp: int, batched: int, num_seqs: int, max_model_len: int,
            limits: dict[str, int] | None = None, max_pixels: int | None = None,
            min_pixels: int | None = None) -> MMProfile | None:
    """What vLLM 0.32 encodes at startup for this encoder, and the memory it holds."""
    limits = limits or {}
    prof = None
    if enc.family == "qwen_vit":
        prof = qwen_vl_profile(enc.model_type, enc.config, enc.preprocessor, batched, num_seqs, max_model_len,
                               limits, max_pixels, min_pixels)
    elif enc.family == "whisper":
        prof = audio_profile(batched, num_seqs, max_model_len, limits)
    if prof is None and all(v == 0 for v in limits.values()) and limits:
        return None
    if prof is None:
        merge = enc.config.get("spatial_merge_size") or 1
        prof = MMProfile(modality=enc.kind, items=1, item_tokens=batched, item_outputs=batched,
                         item_positions=batched * merge ** 2, budget=batched, encoder_peak_bytes=0,
                         output_bytes=0, approximate=True,
                         notes=["the largest item size could not be derived; one item of max_num_batched_tokens "
                                "tokens is assumed"])
    positions = prof.items * prof.item_positions
    prof.encoder_peak_bytes = positions * bytes_per_position(enc, tp)
    prof.output_bytes = prof.items * prof.item_outputs * text_hidden * DTYPE_BYTES
    prof.approximate = prof.approximate or enc.family != "qwen_vit"
    return prof
