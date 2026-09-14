#!/usr/bin/env python3
# Copyright (c) SeasonEngine and contributors.
# Licensed under the MIT License.
# https://github.com/SeasonRealms/SeasonAudio
#
# merge_stable_audio_onnx.py
#
# Consolidate an already-exported Stable Audio ONNX bundle (graph + hundreds of
# per-tensor external-data shards, as produced by `torch.onnx.export`) into the
# runtime layout SeasonEngine expects: one graph file + ONE `<graph>.onnx_data`
# companion per model.
#
# WHY THIS EXISTS (cross-platform / cross-ONNX-Runtime compatibility):
#   Models larger than the 2 GiB protobuf limit cannot be fully inlined, so the
#   big weights must live in an external data file. The mistake that breaks
#   ONNX Runtime >= 1.23 is externalizing the *small Constant tensors that shape
#   inference must read* (Unsqueeze/Reshape/Split axes, ...). When those are
#   file-backed, session creation fails with:
#       [ShapeInferenceError] Cannot parse data from external tensors.
#       Please load external data into raw data for tensor: /Constant_output_0
#   ORT only auto-loads external data for graph *initializers*, never for
#   *Constant-node attribute* tensors. The in-memory variant of this bug was
#   fixed upstream (PR #26263), but the file-backed variant (PR #31169) is still
#   unmerged, so 1.26.0 is affected too.
#
#   The fix is to re-bake the model so every Constant-node attribute tensor and
#   every small initializer stays INLINE in the protobuf, while only large
#   initializers (weights) go to the single `.onnx_data`:
#       onnx.load(..., load_external_data=True)      # pull all shards into RAM
#       onnx.save(..., save_as_external_data=True,   # re-emit
#                 all_tensors_to_one_file=True,
#                 location="<graph>.onnx_data",
#                 size_threshold=1024,
#                 convert_attribute=False)           # <-- keeps Constants inline
#   `convert_attribute=False` is the critical flag: onnx only externalizes
#   initializers, never Constant-node attributes.
#
# This script is pure `onnx` (no torch, no re-export) and platform independent.
#
# USAGE
#   # Non-destructive: write a merged copy next to the source bundle.
#   python merge_stable_audio_onnx.py /path/to/stable-audio-3-medium-onnx
#
#   # Merge into an explicit directory.
#   python merge_stable_audio_onnx.py /path/to/bundle --out-dir /path/to/merged
#
#   # Consolidate the bundle in place (removes the now-redundant shards).
#   python merge_stable_audio_onnx.py /path/to/bundle --in-place
#
#   # Verify an existing model without merging (detects the external-constant bug).
#   python merge_stable_audio_onnx.py --verify-only /path/to/dit.onnx
#
# Requires: pip install "onnx>=1.16"
"""Merge per-tensor ONNX external-data shards into one `.onnx_data` per model."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

DEFAULT_SIZE_THRESHOLD = 1024
PROTOBUF_LIMIT = 2 * 1024 * 1024 * 1024  # hard 2 GiB limit for a single .onnx proto
SKIP_NAMES = {".DS_Store"}


# --------------------------------------------------------------------------- #
# onnx graph helpers
# --------------------------------------------------------------------------- #
def _iter_graphs(graph: Any) -> Iterator[Any]:
    """Yield `graph` and every nested sub-graph (recursively)."""
    import onnx

    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _iter_graphs(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    yield from _iter_graphs(sub)


def _iter_constant_attr_tensors(graph: Any) -> Iterator[Any]:
    """Yield every tensor stored in a node attribute (Constant `value`, etc.)."""
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            for t in attr.tensors:
                yield t


def _is_external(tensor: Any) -> bool:
    import onnx

    return bool(tensor.external_data) or tensor.data_location == onnx.TensorProto.EXTERNAL


def _external_location(tensor: Any) -> str | None:
    for entry in tensor.external_data:
        if entry.key == "location":
            return entry.value
    return None


def collect_tensors(model: Any) -> tuple[list[Any], list[Any]]:
    """Return (initializers, constant_attribute_tensors) across all sub-graphs."""
    initializers: list[Any] = []
    attributes: list[Any] = []
    for graph in _iter_graphs(model.graph):
        initializers.extend(graph.initializer)
        attributes.extend(_iter_constant_attr_tensors(graph))
    return initializers, attributes


def collect_external_locations(model: Any) -> set[str]:
    """Shard file names referenced by any external tensor (relative to the graph)."""
    initializers, attributes = collect_tensors(model)
    locations: set[str] = set()
    for tensor in (*initializers, *attributes):
        if _is_external(tensor):
            loc = _external_location(tensor)
            if loc:
                locations.add(loc)
    return locations


def model_has_external_data(model: Any) -> bool:
    initializers, attributes = collect_tensors(model)
    return any(_is_external(t) for t in (*initializers, *attributes))


# --------------------------------------------------------------------------- #
# consolidation
# --------------------------------------------------------------------------- #
def consolidate_onnx(
    src_graph: Path,
    dst_graph: Path,
    *,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
    convert_attribute: bool = False,
) -> dict[str, Any]:
    """Re-bake `src_graph` into `dst_graph` + a single `<dst_graph>.onnx_data`.

    Constant-node attributes and initializers smaller than `size_threshold` stay
    inline; only large initializers are externalized. Models without external data
    are copied through untouched.
    """
    import onnx

    src_graph = Path(src_graph)
    dst_graph = Path(dst_graph)
    probe = onnx.load(str(src_graph), load_external_data=False)
    if not model_has_external_data(probe):
        dst_graph.parent.mkdir(parents=True, exist_ok=True)
        if src_graph.resolve() != dst_graph.resolve():
            shutil.copy2(src_graph, dst_graph)
        return {
            "path": dst_graph.as_posix(),
            "data_file": None,
            "changed": False,
            "note": "no external data; copied through",
        }

    data_name = f"{dst_graph.name}_data"
    dst_graph.parent.mkdir(parents=True, exist_ok=True)

    # Load first: when consolidating in place the source graph references the very
    # data file we are about to replace, so its bytes must be in memory before then.
    model = onnx.load(str(src_graph), load_external_data=True)

    # The onnx writer refuses to overwrite an existing external-data file.
    stale_data = dst_graph.parent / data_name
    if stale_data.exists():
        stale_data.unlink()

    previous_cwd = os.getcwd()
    try:
        # `location` is resolved relative to the CWD at save time, so run inside
        # the destination directory and use bare file names.
        os.chdir(dst_graph.parent)
        temp_graph_name = f"{dst_graph.name}.tmp"
        if os.path.exists(temp_graph_name):
            os.unlink(temp_graph_name)
        onnx.save(
            model,
            temp_graph_name,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=size_threshold,
            convert_attribute=convert_attribute,
        )
        os.replace(temp_graph_name, dst_graph.name)
    finally:
        os.chdir(previous_cwd)
        del model
        gc.collect()

    return {"path": dst_graph.as_posix(), "data_file": data_name, "changed": True}


# --------------------------------------------------------------------------- #
# verification (mirrors what ORT does at load: shape inference w/o external data)
# --------------------------------------------------------------------------- #
_ITEMSIZE = {
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2,
}


def _tensor_nbytes(tensor: Any) -> int:
    """Best-effort byte size from dims + data_type (works for external tensors too)."""
    count = 1
    for dim in tensor.dims:
        count *= dim
    return count * _ITEMSIZE.get(tensor.data_type, 4)


def verify_model(path: Path, *, deep: bool = False, size_threshold: int = DEFAULT_SIZE_THRESHOLD) -> dict[str, Any]:
    """Check a model does not depend on file-backed constants for shape inference.

    Structural checks (always, fast, no external data loaded):
      * no Constant-node attribute tensor is external (the direct cause of the ORT
        "Cannot parse data from external tensors" load failure), and
      * no initializer that should have stayed inline (byte size below the merge
        `size_threshold`, minus protobuf overhead) is external, and
      * the graph proto is below the 2 GiB protobuf limit.

    With `deep=True` it additionally runs ONNX shape inference without external
    data (mirrors ORT's load-time inference). Deep mode can be slow on large graphs
    and is therefore opt-in.
    """
    import onnx

    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    initializers, attributes = collect_tensors(model)
    external_attrs = sum(1 for t in attributes if _is_external(t))
    external_inits = [t for t in initializers if _is_external(t)]
    # onnx externalizes a tensor when getsizeof(raw_data) >= size_threshold; getsizeof
    # adds ~33 bytes of overhead, so allow that much slack before calling it suspicious.
    inline_bound = max(0, size_threshold - 64)
    external_small_inits = sum(1 for t in external_inits if _tensor_nbytes(t) < inline_bound)
    graph_bytes = path.stat().st_size

    infer_ok = True
    infer_error: str | None = None
    if deep:
        from onnx import shape_inference

        try:
            shape_inference.infer_shapes(model, strict_mode=True)
        except Exception as exc:  # noqa: BLE001 - report any inference failure verbatim
            infer_ok = False
            infer_error = f"{type(exc).__name__}: {exc}"

    ok = (
        external_attrs == 0
        and external_small_inits == 0
        and graph_bytes < PROTOBUF_LIMIT
        and infer_ok
    )

    return {
        "path": path.as_posix(),
        "ok": ok,
        "graph_bytes": graph_bytes,
        "graph_under_2gb": graph_bytes < PROTOBUF_LIMIT,
        "external_constant_attributes": external_attrs,
        "external_initializers": len(external_inits),
        "external_small_initializers": external_small_inits,
        "deep": deep,
        "shape_inference_ok": infer_ok,
        "shape_inference_error": infer_error,
    }


# --------------------------------------------------------------------------- #
# bundle merge
# --------------------------------------------------------------------------- #
def _find_graph_files(bundle_dir: Path) -> list[Path]:
    return sorted(p for p in bundle_dir.rglob("*.onnx") if p.is_file())


def merge_bundle(
    bundle_dir: Path,
    *,
    out_dir: Path | None = None,
    in_place: bool = False,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
    verify: bool = True,
    deep_verify: bool = False,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir).resolve()
    if not bundle_dir.is_dir():
        raise NotADirectoryError(f"Bundle directory not found: {bundle_dir}")

    if in_place:
        target_root = bundle_dir
    else:
        target_root = Path(out_dir).resolve() if out_dir else bundle_dir.parent / f"{bundle_dir.name}-merged"
        if target_root == bundle_dir:
            raise ValueError("--out-dir must differ from the source bundle unless --in-place is used.")
        target_root.mkdir(parents=True, exist_ok=True)

    graphs = _find_graph_files(bundle_dir)
    if not graphs:
        raise FileNotFoundError(f"No .onnx graph files found under {bundle_dir}")

    # Shard files are consumed by consolidation; everything else is copied through.
    shard_paths: set[Path] = set()
    import onnx

    for graph in graphs:
        probe = onnx.load(str(graph), load_external_data=False)
        for loc in collect_external_locations(probe):
            shard_paths.add((graph.parent / loc).resolve())

    results: list[dict[str, Any]] = []
    for graph in graphs:
        rel = graph.relative_to(bundle_dir)
        dst = target_root / rel
        print(f"[merge] {rel} -> {dst}")
        summary = consolidate_onnx(graph, dst, size_threshold=size_threshold)
        if in_place and summary["changed"]:
            # Remove only the shards this graph referenced (never touch other assets).
            removed = 0
            for shard in shard_paths:
                if shard.parent == graph.parent.resolve() and shard.exists() and shard.name != summary["data_file"]:
                    shard.unlink()
                    removed += 1
            summary["removed_shards"] = removed
        if verify:
            summary["verify"] = verify_model(dst, deep=deep_verify, size_threshold=size_threshold)
        results.append(summary)
        del summary
        gc.collect()

    if not in_place:
        # Copy non-graph, non-shard assets (tokenizer, manifest, ...) preserving layout.
        graph_set = {g.resolve() for g in graphs}
        for file in bundle_dir.rglob("*"):
            if not file.is_file() or file.name in SKIP_NAMES:
                continue
            resolved = file.resolve()
            if resolved in graph_set or resolved in shard_paths:
                continue
            rel = file.relative_to(bundle_dir)
            dst = target_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dst)

    return {
        "bundle": bundle_dir.as_posix(),
        "target": target_root.as_posix(),
        "in_place": in_place,
        "size_threshold": size_threshold,
        "models": results,
    }


def _verify_only(paths: list[str], deep: bool = False, size_threshold: int = DEFAULT_SIZE_THRESHOLD) -> int:
    reports = [verify_model(Path(p), deep=deep, size_threshold=size_threshold) for p in paths]
    print(json.dumps({"verify": reports}, indent=2))
    return 0 if all(r["ok"] for r in reports) else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Stable Audio ONNX external-data shards into one .onnx_data per model "
        "(keeps shape-inference constants inline for cross-ONNX-Runtime compatibility).",
    )
    parser.add_argument("bundle_dir", nargs="?", help="Bundle directory containing sa3-m/, same-l/, t5gemma/.")
    parser.add_argument("--out-dir", default=None, help="Destination directory (default: <bundle>-merged). Ignored with --in-place.")
    parser.add_argument("--in-place", action="store_true", help="Consolidate inside the source bundle and delete consumed shards.")
    parser.add_argument("--size-threshold", type=int, default=DEFAULT_SIZE_THRESHOLD, help="Initializers >= this many bytes go external (default: 1024).")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="Skip post-merge shape-inference verification.")
    parser.add_argument("--deep-verify", action="store_true", help="Also run ONNX shape inference (slower) during verification.")
    parser.add_argument("--verify-only", nargs="+", metavar="MODEL", help="Verify existing .onnx file(s) and exit (no merge).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.verify_only:
        return _verify_only(args.verify_only, deep=args.deep_verify, size_threshold=args.size_threshold)

    if not args.bundle_dir:
        print("error: bundle_dir is required (or use --verify-only).", file=sys.stderr)
        return 2

    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError:
        print("error: the `onnx` package is required. Install with: pip install \"onnx>=1.16\"", file=sys.stderr)
        return 3

    summary = merge_bundle(
        Path(args.bundle_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        in_place=args.in_place,
        size_threshold=args.size_threshold,
        verify=args.verify,
        deep_verify=args.deep_verify,
    )

    print(json.dumps(summary, indent=2))
    all_ok = all(m.get("verify", {}).get("ok", True) for m in summary["models"])
    if args.verify and not all_ok:
        print("warning: at least one model failed verification.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
