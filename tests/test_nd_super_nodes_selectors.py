import importlib
import json

MODULE_PATH = "ComfyUI_SaveImageWithMetaDataUniversal.saveimage_unimeta.defs.ext.nd_super_nodes"

# First four entries of a real ``NdSuperLoraLoader`` bundle: two enabled, two disabled.
_REAL_BUNDLE = json.dumps(
    [
        {"lora": "krea/styles/Dark_Fantasy_Red_krea2.safetensors", "enabled": False, "strength": 0.6, "strengthClip": 1},
        {"lora": "krea/styles/V0idv1_k2.safetensors", "enabled": True, "strength": 0.5, "strengthClip": 1},
        {"lora": "krea/styles/krea2_ta86_inkriot_V2.safetensors", "enabled": True, "strength": 0.6, "strengthClip": 1},
        {"lora": "krea/styles/DETAILCORE_FOREVER_BY_STX_V9.safetensors", "enabled": False, "strength": 0.4, "strengthClip": 1},
    ]
)


def _load_module(monkeypatch):
    mod = importlib.import_module(MODULE_PATH)
    monkeypatch.setattr(mod, "resolve_lora_display_names", lambda names: [f"display::{name}" for name in names])
    monkeypatch.setattr(mod, "calc_lora_hash", lambda name, _input: f"hash::{name}")
    mod._NODE_DATA_CACHE.clear()
    return mod


def _read_all(mod, node_id, input_data):
    return (
        mod.get_lora_model_names(node_id, None, None, None, None, input_data),
        mod.get_lora_model_hashes(node_id, None, None, None, None, input_data),
        mod.get_lora_model_strengths(node_id, None, None, None, None, input_data),
        mod.get_lora_clip_strengths(node_id, None, None, None, None, input_data),
    )


def test_json_bundle_captures_only_enabled_entries(monkeypatch):
    mod = _load_module(monkeypatch)
    names, hashes, model_strengths, clip_strengths = _read_all(mod, "1786", ({"lora_bundle": _REAL_BUNDLE},))

    assert names == [
        "display::krea/styles/V0idv1_k2.safetensors",
        "display::krea/styles/krea2_ta86_inkriot_V2.safetensors",
    ]
    assert model_strengths == [0.5, 0.6]
    assert clip_strengths == [1.0, 1.0]
    # Hashes must be computed from the raw bundle name, never the display name.
    assert hashes[0] == "hash::krea/styles/V0idv1_k2.safetensors"


def test_bundle_may_arrive_as_native_list(monkeypatch):
    mod = _load_module(monkeypatch)
    bundle = [
        {"lora": "A.safetensors", "enabled": True, "strength": 0.4, "strengthClip": 0.2},
        {"lora": "B.safetensors", "enabled": True, "strength": 0.9, "strengthClip": 0.1},
    ]
    names, _hashes, model_strengths, clip_strengths = _read_all(mod, "native", ({"lora_bundle": bundle},))

    assert names == ["display::A.safetensors", "display::B.safetensors"]
    assert model_strengths == [0.4, 0.9]
    assert clip_strengths == [0.2, 0.1]


def test_legacy_on_flag_and_strength_two_are_honoured(monkeypatch):
    mod = _load_module(monkeypatch)
    bundle = json.dumps([{"lora": "Legacy.safetensors", "on": True, "strength": 0.35, "strengthTwo": 0.15}])
    names, _hashes, model_strengths, clip_strengths = _read_all(mod, "legacy", ({"lora_bundle": bundle},))

    assert names == ["display::Legacy.safetensors"]
    assert model_strengths == [0.35]
    assert clip_strengths == [0.15]


def test_missing_clip_strength_falls_back_to_model_strength(monkeypatch):
    mod = _load_module(monkeypatch)
    bundle = json.dumps([{"lora": "ModelOnly.safetensors", "enabled": True, "strength": 0.55}])
    _names, _hashes, model_strengths, clip_strengths = _read_all(mod, "clip-fallback", ({"lora_bundle": bundle},))

    assert model_strengths == [0.55]
    assert clip_strengths == [0.55]


def test_entries_without_truthy_flag_are_skipped(monkeypatch):
    mod = _load_module(monkeypatch)
    bundle = json.dumps(
        [
            {"lora": "On.safetensors", "enabled": True, "strength": 1.0, "strengthClip": 1.0},
            {"lora": "Off.safetensors", "enabled": False, "strength": 1.0, "strengthClip": 1.0},
            {"lora": "NoFlag.safetensors", "strength": 1.0, "strengthClip": 1.0},
            {"lora": "None", "enabled": True, "strength": 1.0, "strengthClip": 1.0},
        ]
    )
    names, _hashes, _model_strengths, _clip_strengths = _read_all(mod, "flags", ({"lora_bundle": bundle},))

    assert names == ["display::On.safetensors"]


def test_malformed_and_absent_bundles_yield_empty_lists(monkeypatch):
    mod = _load_module(monkeypatch)

    for node_id, input_data in (
        ("broken", ({"lora_bundle": "[{not json"},)),
        ("not-a-list", ({"lora_bundle": '{"lora": "X.safetensors"}'},)),
        ("absent", ({"model": ["85", 0]},)),
        ("empty", ({"lora_bundle": ""},)),
    ):
        names, hashes, model_strengths, clip_strengths = _read_all(mod, node_id, input_data)
        assert names == []
        assert hashes == []
        assert model_strengths == []
        assert clip_strengths == []


def test_cache_is_invalidated_when_bundle_changes(monkeypatch):
    mod = _load_module(monkeypatch)
    first = json.dumps([{"lora": "First.safetensors", "enabled": True, "strength": 1.0, "strengthClip": 1.0}])
    second = json.dumps([{"lora": "Second.safetensors", "enabled": True, "strength": 1.0, "strengthClip": 1.0}])

    assert mod.get_lora_model_names("cached", None, None, None, None, ({"lora_bundle": first},)) == ["display::First.safetensors"]
    assert mod.get_lora_model_names("cached", None, None, None, None, ({"lora_bundle": first},)) == ["display::First.safetensors"]
    assert mod.get_lora_model_names("cached", None, None, None, None, ({"lora_bundle": second},)) == ["display::Second.safetensors"]


def test_capture_field_list_registers_all_lora_fields():
    mod = importlib.import_module(MODULE_PATH)
    rules = mod.CAPTURE_FIELD_LIST["NdSuperLoraLoader"]

    assert rules[mod.MetaField.LORA_MODEL_NAME]["selector"] is mod.get_lora_model_names
    assert rules[mod.MetaField.LORA_MODEL_HASH]["selector"] is mod.get_lora_model_hashes
    assert rules[mod.MetaField.LORA_STRENGTH_MODEL]["selector"] is mod.get_lora_model_strengths
    assert rules[mod.MetaField.LORA_STRENGTH_CLIP]["selector"] is mod.get_lora_clip_strengths


def _prime_capture(monkeypatch, mod):
    """Point the capture pipeline at a prompt holding one real NdSuperLoraLoader node."""
    cap = importlib.import_module("ComfyUI_SaveImageWithMetaDataUniversal.saveimage_unimeta.capture")

    class DummyOutputs:
        outputs: dict = {}

    class DummyPromptExecuter:
        caches = DummyOutputs()

    class DummyHook:
        current_prompt = {
            "1786": {"class_type": "NdSuperLoraLoader", "inputs": {"model": ["85", 0], "lora_bundle": _REAL_BUNDLE}},
        }
        current_extra_data: dict = {}
        prompt_executer = DummyPromptExecuter()

    monkeypatch.setattr(cap, "NODE_CLASS_MAPPINGS", {"NdSuperLoraLoader": object})
    monkeypatch.setattr(cap, "get_input_data", lambda node_inputs, *args, **kwargs: (node_inputs,))
    monkeypatch.setattr(cap, "hook", DummyHook)
    monkeypatch.setattr(cap, "CAPTURE_FIELD_LIST", mod.CAPTURE_FIELD_LIST)
    mod._NODE_DATA_CACHE.clear()
    return cap


def test_capture_pipeline_emits_one_record_per_enabled_lora(monkeypatch):
    mod = importlib.import_module(MODULE_PATH)
    cap = _prime_capture(monkeypatch, mod)

    loras = cap.Capture.gen_loras(cap.Capture.get_inputs())

    assert sorted(key for key in loras if key.endswith("Model name")) == [f"Lora_{i} Model name" for i in range(2)]
    # Path prefixes are stripped for display, matching every other LoRA loader.
    assert loras["Lora_0 Model name"] == "V0idv1_k2.safetensors"
    assert loras["Lora_0 Strength model"] == 0.5
    assert loras["Lora_0 Strength clip"] == 1.0
    assert loras["Lora_1 Model name"] == "krea2_ta86_inkriot_V2.safetensors"
    assert loras["Lora_1 Strength model"] == 0.6
    # No raw JSON may leak into any recorded value.
    assert not any("lora_bundle" in str(value) for value in loras.values())


def test_extension_registers_a_selector_for_every_lora_field():
    """Loading the extensions must install callable selectors, not a scalar ``field_name`` rule."""
    defs = importlib.import_module("ComfyUI_SaveImageWithMetaDataUniversal.saveimage_unimeta.defs")
    meta = importlib.import_module("ComfyUI_SaveImageWithMetaDataUniversal.saveimage_unimeta.defs.meta")

    snapshot = dict(defs.CAPTURE_FIELD_LIST)
    try:
        defs.load_extensions_only()
        rules = defs.CAPTURE_FIELD_LIST["NdSuperLoraLoader"]
        for field in (
            meta.MetaField.LORA_MODEL_NAME,
            meta.MetaField.LORA_MODEL_HASH,
            meta.MetaField.LORA_STRENGTH_MODEL,
            meta.MetaField.LORA_STRENGTH_CLIP,
        ):
            assert callable(rules[field].get("selector"))
    finally:
        defs.CAPTURE_FIELD_LIST.clear()
        defs.CAPTURE_FIELD_LIST.update(snapshot)


def test_stale_user_json_scalar_rule_cannot_replace_the_selector(monkeypatch, tmp_path):
    """An old ``user_captures.json`` entry pointing at ``lora_bundle`` stays inert.

    The node has no bundled rule, so installs that ran the Metadata Rule Scanner once
    got ``{"field_name": "lora_bundle"}`` written to their local user rules. User JSON
    merges after extensions, so the capture pipeline must keep the callable selector.
    """
    defs = importlib.import_module("ComfyUI_SaveImageWithMetaDataUniversal.saveimage_unimeta.defs")
    meta = importlib.import_module("ComfyUI_SaveImageWithMetaDataUniversal.saveimage_unimeta.defs.meta")
    stale_rules = {
        "NdSuperLoraLoader": {
            "LORA_MODEL_NAME": {"field_name": "lora_bundle"},
            "LORA_MODEL_HASH": {"field_name": "lora_bundle", "format": "calc_lora_hash"},
        },
        # Proves this file was actually merged: an entry only the JSON provides.
        "SomeUserOnlyNode": {"MODEL_NAME": {"field_name": "ckpt_name"}},
    }
    (tmp_path / "user_captures.json").write_text(json.dumps(stale_rules), encoding="utf-8")

    snapshot = dict(defs.CAPTURE_FIELD_LIST)
    monkeypatch.setattr(defs, "resolve_user_rules_dir", lambda: str(tmp_path))
    try:
        defs.load_user_definitions()
        assert "SomeUserOnlyNode" in defs.CAPTURE_FIELD_LIST
        rules = defs.CAPTURE_FIELD_LIST["NdSuperLoraLoader"]
        assert callable(rules[meta.MetaField.LORA_MODEL_NAME].get("selector"))
        assert callable(rules[meta.MetaField.LORA_MODEL_HASH].get("selector"))
        assert callable(rules[meta.MetaField.LORA_STRENGTH_MODEL].get("selector"))
    finally:
        defs.CAPTURE_FIELD_LIST.clear()
        defs.CAPTURE_FIELD_LIST.update(snapshot)
