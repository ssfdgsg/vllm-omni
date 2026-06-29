# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``_derive_supported_tasks`` (issue #4721).

A dedicated TTS pipeline (CosyVoice3: the talker owns the tokenizer, so
``is_comprehension=True``, but it emits codec tokens with ``final_output_type=None``;
code2wav has ``final_output_type="audio"``) must advertise only ``"speech"``, never
``"generate"``. Advertising ``"generate"`` wrongly registers ``/v1/completions`` (and
``/v1/chat/completions``) for a model that only serves ``/v1/audio/speech``, which lets a
raw-text request reach the talker and crash the EngineCore.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.async_omni_engine import _derive_supported_tasks

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _client(is_comprehension: bool) -> SimpleNamespace:
    return SimpleNamespace(is_comprehension=is_comprehension)


def _meta(
    final_output_type: str | None,
    stage_type: str = "llm",
    model_stage: str | None = None,
    final_output: bool = False,
    engine_input_source: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        final_output_type=final_output_type,
        stage_type=stage_type,
        model_stage=model_stage,
        final_output=final_output,
        engine_input_source=[] if engine_input_source is None else engine_input_source,
    )


def test_cosyvoice3_tts_only_advertises_speech_not_generate() -> None:
    # Real CosyVoice3 stage signature (from #4721 diagnostics):
    # talker -> (is_comprehension=True, final_output_type=None);
    # code2wav -> (False, final_output=True, final_output_type="audio", input_source=[0]).
    clients = [_client(True), _client(False)]
    meta = [_meta(None), _meta("audio", final_output=True, engine_input_source=[0])]
    assert _derive_supported_tasks(clients, meta) == ("speech",)


def test_text_only_pipeline_advertises_generate() -> None:
    assert _derive_supported_tasks([_client(True)], [_meta("text")]) == ("generate",)


def test_omni_audio_chat_keeps_generate_with_speech() -> None:
    # A comprehension stage that produces text plus a downstream audio stage
    # (e.g. a thinker-talker omni model) keeps chat/generate routes enabled.
    clients = [_client(True), _client(False)]
    meta = [_meta("text"), _meta("audio")]
    assert _derive_supported_tasks(clients, meta) == ("generate", "speech")


def test_omni_audio_chat_keeps_generate_when_thinker_output_type_is_unknown() -> None:
    # Do not rely solely on final_output_type="text": omni thinker stages can be
    # identified structurally as an additional non-audio final-output branch.
    clients = [_client(True), _client(False), _client(False)]
    meta = [
        _meta(None, final_output=True),
        _meta(None, engine_input_source=[0]),
        _meta("audio", final_output=True, engine_input_source=[1]),
    ]
    assert _derive_supported_tasks(clients, meta) == ("generate", "speech")


def test_audio_diffusion_generation_advertises_generate_not_speech() -> None:
    clients = [_client(False)]
    meta = [_meta("audio", stage_type="diffusion")]
    assert _derive_supported_tasks(clients, meta) == ("generate",)


def test_image_diffusion_generation_advertises_generate() -> None:
    clients = [_client(False)]
    meta = [_meta("image", stage_type="diffusion")]
    assert _derive_supported_tasks(clients, meta) == ("generate",)


def test_video_diffusion_generation_advertises_generate() -> None:
    # LTX-style video diffusion pipelines should keep generate routes enabled.
    clients = [_client(False)]
    meta = [_meta("video", stage_type="diffusion")]
    assert _derive_supported_tasks(clients, meta) == ("generate",)


def test_non_diffusion_image_generation_keeps_generate_default() -> None:
    # MammothModa2 AR->DiT advertises image output without stage_type="diffusion";
    # until tasks are explicit in deploy configs, keep legacy generate routes enabled.
    clients = [_client(True), _client(False)]
    meta = [_meta(None), _meta("image")]
    assert _derive_supported_tasks(clients, meta) == ("generate",)


def test_unknown_non_tts_pipeline_keeps_generate_fallback() -> None:
    assert _derive_supported_tasks([_client(True)], [_meta(None)]) == ("generate",)


def test_empty_pipeline_keeps_generate_default() -> None:
    assert _derive_supported_tasks([], []) == ("generate",)
