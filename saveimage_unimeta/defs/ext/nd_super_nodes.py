"""Provides metadata definitions for the ND Super Nodes custom node pack.

Covers ``NdSuperLoraLoader`` (display name ``ND Super LoRA Loader``). The node keeps
every LoRA it manages in a single JSON string input (``lora_bundle``), so a plain
``field_name`` capture cannot describe it: it would treat the whole JSON blob as one
filename. The selectors below parse the bundle, keep only the entries the node really
applied, and return the four index-aligned lists (names, hashes, model strengths, clip
strengths) that ``Capture`` zips into ``Lora_<n>`` metadata records.

Entry shape produced by the node's frontend::

    {"lora": "sub/dir/name.safetensors", "enabled": true, "strength": 0.5,
     "strengthClip": 1.0, "triggerWords": "", "tag": "General", "autoFetched": false}

Selection mirrors ``NdSuperLoraLoader.load_loras`` so the captured metadata matches what
was actually applied: an entry counts only when ``enabled`` (or the legacy ``on``) is
truthy, ``lora`` names the file, ``strength`` is the model strength, and ``strengthClip``
(or the legacy ``strengthTwo``) is the CLIP strength, defaulting to the model strength.
When the optional ``clip`` input is unconnected the node zeroes every CLIP strength, and
so do these selectors.

Attributes:
    CAPTURE_FIELD_LIST (dict): Maps ``NdSuperLoraLoader`` to its LoRA capture selectors.
"""

import json
import logging
from typing import Any

from ...utils.lora import resolve_lora_display_names
from ..formatters import calc_lora_hash
from ..meta import MetaField

logger = logging.getLogger(__name__)

_BUNDLE_FIELD = "lora_bundle"
_CLIP_FIELD = "clip"

# Cache the parsed+hashed bundle per node id so the two capture passes of a single save
# (before sampler / before save node) do not re-read and re-hash every LoRA file.
_NODE_DATA_CACHE: dict[str, tuple[Any, dict[str, list]]] = {}


def _coerce_float(value: Any) -> float | None:
    """Coerce a bundle value to ``float``.

    Args:
        value: Raw JSON value from a bundle entry.

    Returns:
        float | None: The numeric value, or None when it is missing or not numeric.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _unwrap_bundle(value: Any) -> Any:
    """Strip widget wrappers from a bundle value.

    A one-element list only counts as a wrapper when its item cannot be a bundle entry
    itself (a JSON string or a ``{"__value__": ...}`` box). A native bundle holding a
    single entry must stay a list, or ``_enabled_entries`` would discard it.

    Args:
        value: Raw ``lora_bundle`` input, possibly wrapped by the frontend or by an
            output cache as a single-element list or ``{"__value__": ...}``.

    Returns:
        Any: The innermost bundle value.
    """
    while isinstance(value, list | tuple) and len(value) == 1:
        item = value[0]
        if isinstance(item, dict) and "__value__" not in item:
            break
        value = item
    if isinstance(value, dict) and "__value__" in value:
        return _unwrap_bundle(value["__value__"])
    return value


def _clip_available(batch: Any) -> bool:
    """Report whether the node's optional ``clip`` input is connected.

    ``NdSuperLoraLoader.load_loras`` applies ``strength_clip = 0`` when no CLIP is
    supplied, so a model-only workflow must not report CLIP strengths.

    Args:
        batch: Resolved node inputs from the capture pipeline (``input_data[0]``).

    Returns:
        bool: True when a CLIP value is present and not an unresolved ``(None,)`` marker.
    """
    if not isinstance(batch, dict):
        return False
    value = batch.get(_CLIP_FIELD)
    if isinstance(value, list | tuple):
        value = value[0] if value else None
    return value is not None


def _enabled_entries(bundle: Any) -> list[tuple[str, float, float]]:
    """Parse ``lora_bundle`` into ``(name, strength_model, strength_clip)`` tuples.

    Disabled and blank entries are dropped so metadata reflects the LoRAs the node
    actually loaded. Malformed payloads yield an empty list: capture must never fail
    an image save because a node wrote something unexpected.

    Args:
        bundle: Raw ``lora_bundle`` input (JSON string, list, or widget wrapper).

    Returns:
        list[tuple[str, float, float]]: Enabled entries in bundle (load) order.
    """
    bundle = _unwrap_bundle(bundle)
    if isinstance(bundle, str):
        if not bundle.strip().startswith("["):
            return []
        try:
            bundle = json.loads(bundle)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("[ND Super Meta] Ignoring unparsable lora_bundle: %r", exc)
            return []
    if not isinstance(bundle, list | tuple):
        return []

    entries: list[tuple[str, float, float]] = []
    for item in bundle:
        if not isinstance(item, dict):
            continue
        # Same flag lookup as NdSuperLoraLoader.load_loras: no truthy flag means not applied.
        if not bool(item.get("enabled", item.get("on", False))):
            continue
        name = item.get("lora") or item.get("name") or item.get("model")
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name or name == "None":
            continue
        strength_model = _coerce_float(item.get("strength"))
        if strength_model is None:
            strength_model = 1.0
        strength_clip = _coerce_float(item.get("strengthClip"))
        if strength_clip is None:
            strength_clip = _coerce_float(item.get("strengthTwo"))
        if strength_clip is None:
            strength_clip = strength_model
        entries.append((name, strength_model, strength_clip))
    return entries


def _build_result(entries: list[tuple[str, float, float]]) -> dict[str, list]:
    """Resolve display names and hashes for the enabled entries.

    All four lists stay index-aligned: a failed hash lookup is recorded as None for
    that slot rather than dropped, so names and strengths never shift out of step.

    Args:
        entries: ``(name, strength_model, strength_clip)`` tuples in bundle order.

    Returns:
        dict[str, list]: ``names``, ``hashes``, ``model_strengths`` and ``clip_strengths``.
    """
    result: dict[str, list] = {"names": [], "hashes": [], "model_strengths": [], "clip_strengths": []}
    if not entries:
        return result

    raw_names = [entry[0] for entry in entries]
    display_names = resolve_lora_display_names(raw_names)
    for raw_name, display_name, (_, strength_model, strength_clip) in zip(raw_names, display_names, entries):
        try:
            hash_value = calc_lora_hash(raw_name, [])
        except Exception as exc:  # pragma: no cover - hashing must never abort capture
            logger.debug("[ND Super Meta] Hash lookup failed for %s: %r", raw_name, exc)
            hash_value = None
        result["names"].append(display_name)
        result["hashes"].append(hash_value)
        result["model_strengths"].append(strength_model)
        result["clip_strengths"].append(strength_clip)
    return result


def _cache_key(bundle: Any) -> Any:
    """Return a hashable signature for a bundle value.

    Args:
        bundle: Raw ``lora_bundle`` input.

    Returns:
        Any: A hashable value that only compares equal for identical bundles.
    """
    if isinstance(bundle, str):
        return bundle
    try:
        return json.dumps(bundle, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(bundle)


def _get_lora_data_from_node(node_id: Any, input_data: Any) -> dict[str, list]:
    """Return parsed LoRA data for one node, cached on the bundle and CLIP state.

    Args:
        node_id: Unique id of the node in the prompt; used as the cache key.
        input_data: Capture pipeline tuple whose first element maps resolved inputs.

    Returns:
        dict[str, list]: ``names``, ``hashes``, ``model_strengths`` and ``clip_strengths``.
            Clip strengths are zeroed when no CLIP input is connected.
    """
    batch = input_data[0] if input_data and input_data[0] else None
    bundle = batch.get(_BUNDLE_FIELD) if isinstance(batch, dict) else None
    clip_available = _clip_available(batch)
    key = (_cache_key(bundle), clip_available)
    cache_slot = str(node_id)
    cached = _NODE_DATA_CACHE.get(cache_slot)
    if cached and cached[0] == key:
        return cached[1]

    data = _build_result(_enabled_entries(bundle))
    if not clip_available:
        data["clip_strengths"] = [0.0] * len(data["clip_strengths"])
    _NODE_DATA_CACHE[cache_slot] = (key, data)
    return data


# Selectors (note: *args[-1] is the input_data structure from the capture pipeline)
def get_lora_model_names(node_id: Any, *args: Any) -> list[str]:
    """Selector returning the enabled LoRA display names for a node."""
    return _get_lora_data_from_node(node_id, args[-1])["names"]


def get_lora_model_hashes(node_id: Any, *args: Any) -> list[str | None]:
    """Selector returning the enabled LoRA hashes for a node."""
    return _get_lora_data_from_node(node_id, args[-1])["hashes"]


def get_lora_model_strengths(node_id: Any, *args: Any) -> list[float]:
    """Selector returning the enabled LoRA model strengths for a node."""
    return _get_lora_data_from_node(node_id, args[-1])["model_strengths"]


def get_lora_clip_strengths(node_id: Any, *args: Any) -> list[float]:
    """Selector returning the enabled LoRA CLIP strengths for a node."""
    return _get_lora_data_from_node(node_id, args[-1])["clip_strengths"]


CAPTURE_FIELD_LIST = {
    "NdSuperLoraLoader": {
        MetaField.LORA_MODEL_NAME: {"selector": get_lora_model_names},
        MetaField.LORA_MODEL_HASH: {"selector": get_lora_model_hashes},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_lora_model_strengths},
        MetaField.LORA_STRENGTH_CLIP: {"selector": get_lora_clip_strengths},
    },
}
