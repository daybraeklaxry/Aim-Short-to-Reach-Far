"""Corrected released Native CEM on the unchanged main query cohort."""
import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import time
from eval_spec import BASE, TASKS, ARMS, read, write, identity, result_stem


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_shared(root, protocol):
    main = Path(protocol['main_root'])
    assert (main / 'protocol.json').read_bytes() == (root / 'main_protocol.json').read_bytes()
    for name, path in protocol['source_paths'].items():
        assert Path(path).read_bytes() == (root / 'frozen_code' / name).read_bytes(), path
    shared = root / 'frozen_code/shared_api'
    sys.path.insert(0, str(shared))
    api = module_from_file('native_shared_confirmation', shared / 'run_confirm.py')
    return api, api.load_runtime(main, read(root / 'main_protocol.json'))


def rollout(ew, api, context, native, task, row, study, stage, root, protocol):
    from verify import validate_result
    np = ew.np
    arm = native.arm
    stem = result_stem(root, task, stage, row, arm)
    previous = api.load_pair(stem, np)
    if previous is not None:
        validate_result(*previous, row, study, stage, protocol, task, arm)
        native.restore_rng(previous[1]['native_rng_after'])
        return previous[0]
    main = Path(protocol['main_root'])
    metadata, cached = api.shared_cache(ew, context, task, row, stage, main, allow_create=False)
    env, obs, info, true_goal = ew.runtime._make_live(context, row)
    actions, flags, terms, truncs, decisions, currents, goals, plans, normalized, trajectory = ([] for _ in range(10))
    started = time.perf_counter()
    try:
        assert ew.equal_state(ew.physical_state(task, env), metadata['initial'])
        if task == 'cube':
            assert not env.unwrapped._visualize_info
        np.testing.assert_array_equal(env.action_space.low, metadata['action_low'])
        np.testing.assert_array_equal(env.action_space.high, metadata['action_high'])
        success = bool(ew.runtime._success_now(task, obs, info, true_goal, False, False))
        assert success == metadata['initial_success']
        initial_success, stopped, renders = success, False, 0
        native.begin(row, cached)
        rng_before = native.rng_state()
        trajectory.append(metadata['initial'])
        while not success and not stopped and len(actions) < study['budget']:
            step = len(actions)
            began = time.perf_counter()
            raw, decision, inputs = native.choose(None, step)
            planning_seconds = time.perf_counter() - began
            assert raw.shape == (ARMS[arm], len(metadata['action_low'])) and np.isfinite(raw).all()
            currents.append(inputs['current'].copy())
            goals.append(native.goal.reshape(-1).copy())
            plans.append(native.last_raw_plan.copy())
            normalized.append(native.last_normalized_plan.reshape(5, -1).copy())
            decision.update(t=step, planning_seconds=planning_seconds,
                            raw_action=raw.tolist(), clipped_components=0)
            for index, action in enumerate(raw):
                obs, _, terminated, truncated, info = env.step(action.copy())
                success = bool(ew.runtime._success_now(task, obs, info, true_goal, False, False))
                actions.append(action.copy())
                flags.append(success); terms.append(bool(terminated)); truncs.append(bool(truncated))
                stopped = bool((terminated or truncated) and not success)
                trajectory.append(ew.physical_state(task, env))
                continuing = not success and not stopped and len(actions) < study['budget']
                renders += native.completed_primitive(env, index, continuing)
                if not continuing:
                    break
            decision['executed_primitive_count'] = len(actions) - step
            decisions.append(decision)
        dimension, action_dim = cached['goal'].size, len(metadata['action_low'])
        result = dict(identity=identity(row), record_id=row['record_id'], episode=int(row['episode']),
            task=task, stage=stage, arm=arm, label=f'{arm}_seed42', training_seed=None, cem_seed=42,
            budget=study['budget'], goal_offset=study['horizon'], primitive_steps=len(actions),
            initial_success=initial_success, success=success, terminal_without_success=stopped,
            initial_physics_verified=True, shared_cache=str(main / 'cache' / task / f"{stage}_{row['record_id']}"),
            main_frozen_utc=protocol['main_frozen_utc'], extension_frozen_utc=protocol['frozen_utc'],
            checkpoint='unchanged context model.official; strict original LeWM checkpoint loading',
            normalization_source=native.normalization_source, normalization_scope='full original dataset',
            action_low=metadata['action_low'], action_high=metadata['action_high'],
            action_handling='StandardScaler inverse_transform; no candidate or raw-action clipping',
            initial_history_len=1, rolling_prediction_history=3, replan_steps=ARMS[arm],
            precision='fp32 without autocast', decisions=decisions, physical_trajectory=trajectory,
            trajectory_semantics='Initial physical state then each actual primitive-step state; action plans stored separately',
            shared_cache_reused=True, shared_cache_encoder_calls=0,
            native_macro_renders=renders, encoding_calls_in_cost=60 * len(decisions),
            predictor_calls=150 * len(decisions),
            planning_seconds=sum(d['planning_seconds'] for d in decisions),
            elapsed_seconds=time.perf_counter()-started, compute_device=str(context['device']),
            renderer=os.environ.get('MUJOCO_GL'), renderer_device=os.environ.get('MUJOCO_EGL_DEVICE_ID'))
        arrays = dict(actions=np.asarray(actions, np.float32).reshape(-1, action_dim),
            success=np.asarray(flags, bool), terminated=np.asarray(terms, bool), truncated=np.asarray(truncs, bool),
            decision_latents=np.asarray(currents, np.float32).reshape(-1, dimension),
            target_latents=np.asarray(goals, np.float32).reshape(-1, dimension),
            planned_raw_actions=np.asarray(plans, np.float32).reshape(-1, 25, action_dim),
            planned_normalized_blocks=np.asarray(normalized, np.float32).reshape(-1, 5, 5*action_dim),
            native_initial_rgb=native.initial_history.copy(), goal_rgb=cached['goal_rgb'].copy(),
            native_initial_past=native.initial_past.copy(),
            native_action_mean=native.mean, native_action_std=native.std,
            native_rng_before=rng_before, native_rng_after=native.rng_state())
        validate_result(result, arrays, row, study, stage, protocol, task, arm)
        api.save_arrays(stem.with_suffix('.npz'), np, **arrays)
        write(stem.with_suffix('.json'), result)
        print(f"{task} {stage} {row['record_id']} {arm} seed42 success={success} steps={len(actions)}", flush=True)
        return result
    finally:
        env.close()


def run(args):
    root = args.root.resolve()
    protocol = read(root / 'protocol.json')
    study = protocol['studies'][args.task]
    stop = len(study[args.stage]) if args.stop_index is None else args.stop_index
    assert 0 <= args.start_index < stop <= len(study[args.stage])
    api, ew = load_shared(root, protocol)
    from verify import verify_one, verify_segment
    if args.stage == 'confirm':
        for profile_row in study['profile']:
            verify_one(root, protocol, args.task, 'profile', profile_row, args.arm, api)
        marker = read(Path(protocol['main_root']) / 'cache' / f'{args.task}_confirm_complete.json')
        assert marker == dict(complete=True, task=args.task, stage='confirm', count=64,
                              frozen_utc=protocol['main_frozen_utc'])
    torch = ew.torch
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(protocol['cuda_memory_fraction'], device)
    context, _ = ew.runtime._context(args.task, study['horizon'], 32, 50, device)
    context['model'].eval().requires_grad_(False)
    versions = {key: importlib.metadata.version(key) for key in
                ('torch', 'numpy', 'mujoco', 'dm-control', 'scikit-learn')}
    if args.task == 'reacher':
        assert versions['dm-control'] == '1.0.43' and versions['mujoco'] == '3.10.0'
    if args.task in ('cube', 'reacher'):
        assert os.environ.get('MUJOCO_GL') == os.environ.get('PYOPENGL_PLATFORM') == 'osmesa'
    from native_adapter import NativeAdapter
    native = NativeAdapter(ew, context, args.task, protocol, root, args.arm)
    runtime = dict(versions=versions, renderer=os.environ.get('MUJOCO_GL'), device=str(device),
        extension_frozen_utc=protocol['frozen_utc'], main_frozen_utc=protocol['main_frozen_utc'],
        start_index=args.start_index, stop_index=stop, arm=args.arm, seed=42)
    path = root / 'results' / args.task / f'{args.stage}_{args.arm}_{args.start_index}_{stop}_runtime.json'
    if path.exists():
        assert read(path) == runtime, 'Segment runtime changed during resume'
    else:
        write(path, runtime)
    if args.start_index:
        previous_row = study[args.stage][args.start_index - 1]
        _, arrays = verify_one(root, protocol, args.task, args.stage, previous_row, args.arm, api)
        native.restore_rng(arrays['native_rng_after'])
    with torch.inference_mode():
        for row in study[args.stage][args.start_index:stop]:
            rollout(ew, api, context, native, args.task, row, study, args.stage, root, protocol)
    verify_segment(root, protocol, args.task, args.stage, args.arm, args.start_index, stop, api)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=BASE / 'hwm_reproduction_audit/native_control')
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--stage', choices=('profile', 'confirm'), required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--stop-index', type=int)
    run(parser.parse_args())
