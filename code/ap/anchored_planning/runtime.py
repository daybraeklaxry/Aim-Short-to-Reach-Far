"""Portable asset I/O and evaluator wiring around the frozen numerical kernels.

This runner performs new simulations. The supplied result tables remain the
record of the completed experiment; no simulation is implied by CPU analysis.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

from .analysis import read_json, write_json

PACKAGES = ("numpy", "torch", "torchvision", "transformers", "tokenizers", "einops",
            "gymnasium", "stable-worldmodel", "scikit-learn", "h5py", "hdf5plugin",
            "mujoco", "dm-control", "pygame", "pymunk", "ogbench")


def assets_from_config(path, task):
    path = Path(path).resolve()
    config = read_json(path)
    asset = dict(config["tasks"][task])
    for name in ("checkpoint", "dataset", "latent_cache"):
        asset[name] = (path.parent / Path(asset[name]).expanduser()).resolve()
    return asset


def versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def configure_runtime():
    # The paper used Linux software rendering. Set these before importing the
    # environment; an explicitly chosen renderer remains visible in run.json.
    for key, value in dict(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
        GALLIUM_DRIVER="llvmpipe", LP_NUM_THREADS="1", SDL_VIDEODRIVER="dummy",
        PYGAME_HIDE_SUPPORT_PROMPT="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2").items():
        os.environ.setdefault(key, value)
    import torch
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def check_simulator_versions(task):
    expected = {"mujoco": "3.10.0", "dm-control": "1.0.43"} if task == "reacher" else {
        "mujoco": "3.12.0", "dm-control": "1.0.44"}
    actual = versions()
    wrong = {name: {"required": value, "installed": actual[name]}
             for name, value in expected.items() if actual[name] != value}
    if wrong:
        raise ValueError(f"Simulator versions differ from the frozen {task} runtime: {wrong}. See requirements/.")


def read_latents(path):
    import numpy as np
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return payload["latents"].numpy()


def inspect_assets(args, asset):
    import hdf5plugin  # Registers codecs used by the original HDF5 datasets.
    import h5py
    report = dict(task=args.task, dependencies=versions(), files={})
    for name in ("checkpoint", "dataset", "latent_cache"):
        path = asset[name]
        report["files"][name] = dict(path=str(path), exists=path.is_file(),
            bytes=path.stat().st_size if path.is_file() else None)
    if not asset["checkpoint"].is_file():
        raise FileNotFoundError(f"Missing checkpoint: {asset['checkpoint']}")
    with h5py.File(asset["dataset"], "r") as handle:
        state_key = asset.get("state_key", "state" if args.task == "pusht" else "observation")
        required = ("pixels", "action", "ep_len", "ep_offset", state_key)
        report["datasets"] = {key: dict(shape=list(handle[key].shape), dtype=str(handle[key].dtype)) for key in required}
        total = len(handle["pixels"])
        if len(handle["action"]) != total or len(handle[state_key]) != total:
            raise ValueError("Pixels, states, and actions must use the original shared row order")
    if asset["latent_cache"].is_file():
        latents = read_latents(asset["latent_cache"])
        report["latents"] = dict(shape=list(latents.shape), dtype=str(latents.dtype))
        if latents.shape != (total, 192):
            raise ValueError(f"Cache has {latents.shape}; expected {(total, 192)}")
    else:
        report["cache_status"] = "missing: use encode-cache or provide the recorded cache"
    print(json.dumps(report, indent=2))


def encode_cache(args, asset):
    import hdf5plugin
    import h5py
    import numpy as np
    import torch
    from .model import encode_retrieval, load_model
    destination = asset["latent_cache"]
    if destination.suffix != ".npy":
        raise ValueError("Set latent_cache to a new .npy path for encode-cache; existing .pt caches can be read")
    if destination.exists():
        raise ValueError(f"Cache already exists: {destination}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    model = load_model(asset["checkpoint"], device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    started = time.perf_counter()
    with h5py.File(asset["dataset"], "r") as handle:
        pixels = handle["pixels"]
        total = len(pixels)
        cache = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(total, 192))
        with torch.inference_mode():
            for start in range(0, len(pixels), args.batch_size):
                stop = min(start + args.batch_size, len(pixels))
                cache[start:stop] = encode_retrieval(model, np.asarray(pixels[start:stop]), device).astype(np.float16)
                if start == 0 or stop == len(pixels) or start // args.batch_size % 100 == 0:
                    print(json.dumps(dict(encoded_rows=stop, total_rows=len(pixels))), flush=True)
        cache.flush()
    del cache
    partial.replace(destination)
    write_json(destination.with_suffix(".json"), dict(task=args.task, rows=total,
        latent_dim=192, storage_dtype="float16", encode_batch_size=args.batch_size,
        retrieval_autocast="bfloat16 on CUDA; float32 on CPU", elapsed_seconds=time.perf_counter()-started,
        dependencies=versions()))


class WorkCounter:
    def __init__(self, model):
        self.calls = self.candidates = 0
        self.handle = model.predictor.register_forward_pre_hook(self.observe)

    def observe(self, module, args):
        self.calls += 1
        self.candidates += int(args[0].shape[0])

    def reset(self):
        self.calls = self.candidates = 0


def evaluate(args, asset):
    import hdf5plugin
    import h5py
    import numpy as np
    import torch
    from . import environments
    from .action_domain import prepare_candidates, project
    from .evaluation_helpers import apply_prelude
    from .gaussian_cem import query_seed
    from .lifecycle import install_cube_renderer_cleanup
    from .model import encode_retrieval, load_model
    from .native_adapter import NativeAdapter
    from .observation_bank import ObservationBank
    from .planners import TargetGaussian, TargetRank

    check_simulator_versions(args.task)
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Require shards >= 1 and 0 <= shard-index < shards")
    if args.limit is not None and not 1 <= args.limit <= 128:
        raise ValueError("limit must be between 1 and 128")
    study = args.study.replace("-", "_")
    protocol = read_json(args.root / "protocols" / f"{study}.json")
    selected = protocol["studies"][args.task]
    rows = selected["confirm"][:args.limit]
    train = np.asarray(protocol["bank_train_episodes"][args.task], np.int64)
    if set(train) & {q["episode"] for q in selected["confirm"]}:
        raise ValueError("Query episodes overlap the frozen training memory")
    sources = {(row["record_id"], row["condition"]): row["source"]
               for row in protocol["prefix_sources"] if row["task"] == args.task}
    tag = f"{args.task}_shard{args.shard_index}of{args.shards}"
    args.output.mkdir(parents=True, exist_ok=True)
    outcomes_path = args.output / f"outcomes_{tag}.jsonl"
    if outcomes_path.exists():
        raise ValueError(f"Outcome file exists; choose a new output directory: {outcomes_path}")
    device = torch.device(args.device)
    model = load_model(asset["checkpoint"], device)
    context = dict(task=args.task, device=device, model=model)
    native = NativeAdapter(None, context, args.task,
        {"native_action_stats": read_json(args.root / "protocols/scalers.json")}, args.root)
    counter = WorkCounter(native.model)
    facade = SimpleNamespace(np=np, runtime=environments)
    if args.task == "cube":
        install_cube_renderer_cleanup(facade)
    write_json(args.output / f"run_{tag}.json", dict(study=study, task=args.task,
        shard_index=args.shard_index, shards=args.shards, limit=args.limit,
        device=str(device), dependencies=versions(), numeric_protocol=protocol["numeric_protocol"],
        renderer={key: os.environ.get(key) for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "GALLIUM_DRIVER")},
        assets={key: str(value) for key, value in asset.items()},
        status="new simulation; not a replay of released outcomes"))
    completed = 0
    try:
        with h5py.File(asset["dataset"], "r") as handle, outcomes_path.open("x", encoding="utf-8") as stream:
            lengths = np.asarray(handle["ep_len"], np.int64)
            offsets = np.asarray(handle["ep_offset"], np.int64)
            latents = read_latents(asset["latent_cache"])
            if latents.shape != (len(handle["pixels"]), 192) or int(lengths.sum()) != len(latents):
                raise ValueError("Dataset and cache rows do not match")
            bank = ObservationBank(dict(lengths=lengths, offsets=offsets), latents, train,
                                   5, np.ones(3, np.float32), device)
            scorer = TargetRank(native, handle["pixels"])
            gaussian = TargetGaussian(native, handle["pixels"])
            state_key = asset.get("state_key", "state" if args.task == "pusht" else "observation")
            for ordinal, row in enumerate(rows):
                if ordinal % args.shards != args.shard_index:
                    continue
                states = {int(index): np.asarray(handle[state_key][index], np.float32)
                          for index in (row["global_start"], row["global_goal"])}
                goal_rgb = np.ascontiguousarray(handle["pixels"][row["global_goal"]])
                goal = encode_retrieval(model, goal_rgb[None], device)
                for condition in protocol["conditions"]:
                    prefix_index = sources[row["record_id"], condition]
                    prefix = np.empty((0, native.action_dim), np.float32) if prefix_index is None else np.asarray(
                        handle["action"][prefix_index:prefix_index+5], np.float32)
                    first_rgb = first_prelude = None
                    for arm in protocol["arms"]:
                        env, obs, info, true_goal = environments._make_live({"task": args.task, "arrays": {"states": states}}, row)
                        try:
                            low, high = env.action_space.low.copy(), env.action_space.high.copy()
                            effective_prefix, _ = project(prefix, low, high)
                            obs, info, prelude, _ = apply_prelude(facade, args.task, env, obs, info, true_goal, effective_prefix)
                            rgb = np.ascontiguousarray(env.render())
                            if first_rgb is None:
                                first_rgb, first_prelude = rgb.copy(), prelude.copy()
                            else:
                                np.testing.assert_array_equal(rgb, first_rgb, err_msg="Paired starting observations changed")
                                if prelude != first_prelude:
                                    raise ValueError("Paired prefix outcomes changed")
                            gaussian.set_action_bounds(low, high)
                            generator = torch.Generator(device=device).manual_seed(query_seed(args.task, ordinal))
                            success, stopped = prelude["success"], prelude["terminal_without_success"]
                            actions, decisions = [], []
                            started = time.perf_counter()
                            with torch.inference_mode():
                                while not success and not stopped and len(actions) < selected["budget"]:
                                    step = len(actions)
                                    if step:
                                        rgb = np.ascontiguousarray(env.render())
                                    current = encode_retrieval(model, rgb[None], device)
                                    remaining = max(5, int(row["horizon"]) - step)
                                    support, costs = bank.nearest_topk(current, goal, remaining, 8)
                                    support = support[0]
                                    counter.reset()
                                    if arm in ("direct", "ap_observed", "ap_final"):
                                        raw = np.stack([np.asarray(handle["action"][int(i):int(i)+5], np.float32) for i in support])
                                        effective, normalized, projection = prepare_candidates(raw, low, high, native.scaler)
                                        rank, detail = scorer.choose(arm, rgb, support, normalized, row["global_goal"])
                                        proposed = effective[rank]
                                        detail.update(selected_rank=rank, action_projection=projection)
                                        expected_calls, expected_blocks = (0, 0) if arm == "direct" else (1, 8)
                                    else:
                                        proposed, detail = gaussian.choose(arm, rgb, support, generator, row["global_goal"])
                                        expected_calls, expected_blocks = 31, 9002
                                    assert (counter.calls, counter.candidates) == (expected_calls, expected_blocks)
                                    detail.update(t=step, remaining=remaining, retrieved_sources=support.tolist(),
                                        retrieval_cost=costs[0].tolist(), predictor_calls=counter.calls,
                                        predictor_candidate_blocks=counter.candidates)
                                    for action in proposed:
                                        obs, _, terminated, truncated, info = env.step(action.copy())
                                        success = bool(environments._success_now(args.task, obs, info, true_goal, False, False))
                                        stopped = bool((terminated or truncated) and not success)
                                        actions.append(action.tolist())
                                        if success or stopped or len(actions) >= selected["budget"]:
                                            break
                                    detail["executed_actions"] = len(actions) - step
                                    decisions.append(detail)
                            compact = dict(study=study, task=args.task, identity=row, condition=condition, arm=arm,
                                completed=True, success=bool(success), primitive_steps=len(actions), decisions=len(decisions),
                                budget=selected["budget"], horizon=row["horizon"], prelude=prelude,
                                predictor_calls=sum(d["predictor_calls"] for d in decisions),
                                predictor_candidate_blocks=sum(d["predictor_candidate_blocks"] for d in decisions),
                                source_prediction_misses=0, optimizer_seed=query_seed(args.task, ordinal),
                                query_ordinal=ordinal, profile_censored=False, terminal_without_success=bool(stopped),
                                stop_reason="success" if success else "termination" if stopped else "budget",
                                elapsed_seconds=time.perf_counter()-started)
                            stem = args.output / "traces" / args.task / f"{row['record_id']}_{condition}_{arm}.json"
                            write_json(stem, dict(compact=compact, actions=actions, decisions=decisions))
                            stream.write(json.dumps(compact, allow_nan=False) + "\n")
                            stream.flush()
                            completed += 1
                            print(json.dumps(dict(task=args.task, query=ordinal, condition=condition, arm=arm,
                                success=bool(success), steps=len(actions), completed_outcomes=completed)), flush=True)
                        finally:
                            env.close()
                scorer.encodings.clear()
                gaussian.targets.clear()
    finally:
        counter.handle.remove()
    expected = sum(i % args.shards == args.shard_index for i in range(len(rows))) * len(protocol["conditions"]) * len(protocol["arms"])
    if completed != expected:
        raise ValueError(f"Incomplete shard: {completed}/{expected}")
    write_json(args.output / f"summary_{tag}.json", dict(completed=True, outcomes=completed,
        full_task_cohort=args.limit is None and args.shards == 1))


def run(args):
    asset = assets_from_config(args.assets, args.task)
    if args.command == "inspect-assets":
        inspect_assets(args, asset)
        return
    configure_runtime()
    if args.command == "encode-cache":
        encode_cache(args, asset)
    else:
        evaluate(args, asset)
