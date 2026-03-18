#!/usr/bin/env python3
"""
Evaluate Controllers
--------------------

Utility script to prewarm the ecology engine and compare multiple controllers
(e.g., ManualDriver vs PredictiveController). Outputs summary metrics and can
optionally export snapshots for later visualization.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Type

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine import core
from controllers.manual_driver import ManualDriver
from controllers.predictive_controller import PredictiveController

CONTROLLERS: Dict[str, Type] = {
    'manual': ManualDriver,
    'predictive': PredictiveController,
}

def instrument_system(system: core.Ecology2DSystem, event_log: List[dict]):
    original_apply = system.apply_external_input

    def wrapped(event: core.ExternalInput):
        applied = original_apply(event)
        event_log.append({
            'time_step': int(system.time_step),
            'input': event.to_dict(),
            'applied': bool(applied),
        })
        return applied

    system.apply_external_input = wrapped  # type: ignore


def run_sequence(cfg: core.Ecology2DConfig, controller_name: str, prewarm_steps: int,
                 run_steps: int, step_interval: float, log_events: bool):
    controller_cls = CONTROLLERS[controller_name]
    controller = controller_cls(cfg, step_interval=step_interval)
    event_log: List[dict] = []
    if log_events:
        instrument_system(controller.system, event_log)
    start = time.time()

    # warmup阶段：仅世界自由演化，不让控制器注入动作
    prewarm_start = time.time()
    for _ in range(prewarm_steps):
        controller.system.step()
    prewarm_end = time.time()
    prewarm_output = controller.system.get_system_output(
        metadata={'phase': 'prewarm_end', 'controller': controller_name}
    )

    # measured阶段：控制器正式介入
    measured_start = time.time()
    controller.run(total_steps=run_steps)
    measured_end = time.time()
    measured_output = controller.system.get_system_output(
        metadata={'phase': 'measured_end', 'controller': controller_name}
    )

    end = time.time()
    print(
        f"[{controller_name}] prewarm={prewarm_steps} ({prewarm_end-prewarm_start:.2f}s), "
        f"measured={run_steps} ({measured_end-measured_start:.2f}s), total={end-start:.2f}s"
    )
    return {
        'system': controller.system,
        'event_log': event_log,
        'prewarm_output': prewarm_output,
        'measured_output': measured_output,
    }

def summarize(prewarm_output: core.SystemOutput, measured_output: core.SystemOutput, label: str):
    pre_emergence = prewarm_output.emergence or {}
    pre_stats = prewarm_output.stats or {}
    ms_emergence = measured_output.emergence or {}
    ms_stats = measured_output.stats or {}

    div_gain = float(ms_emergence.get('genetic_diversity', 0.0)) - float(pre_emergence.get('genetic_diversity', 0.0))
    emer_gain = float(ms_emergence.get('emergence_score', 0.0)) - float(pre_emergence.get('emergence_score', 0.0))

    print(
        f"Summary ({label}): "
        f"prewarm(div={pre_emergence.get('genetic_diversity',0):.3f}, "
        f"emer={pre_emergence.get('emergence_score',0):.3f}, "
        f"alive={pre_stats.get('alive_particles',0)}, "
        f"energy={pre_stats.get('total_energy',0):.1f}) | "
        f"measured(div={ms_emergence.get('genetic_diversity',0):.3f}, "
        f"emer={ms_emergence.get('emergence_score',0):.3f}, "
        f"alive={ms_stats.get('alive_particles',0)}, "
        f"energy={ms_stats.get('total_energy',0):.1f}) | "
        f"gain(div={div_gain:.3f}, emer={emer_gain:.3f})"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate controllers on shared config")
    parser.add_argument('--world-size', type=int, default=80)
    parser.add_argument('--particles', type=int, default=1200)
    parser.add_argument('--prewarm', type=int, default=1000)
    parser.add_argument('--run', type=int, default=500)
    parser.add_argument('--controllers', nargs='+', default=['manual','predictive'])
    parser.add_argument('--step-interval', type=float, default=0.0)
    parser.add_argument('--log-events', action='store_true', help='Record controller inputs to JSONL files')
    parser.add_argument('--log-prefix', type=str, default='controller_events', help='Prefix for JSONL log files')
    args = parser.parse_args()

    cfg = core.Ecology2DConfig(world_size=args.world_size, n_particles=args.particles)

    for ctrl in args.controllers:
        if ctrl not in CONTROLLERS:
            print(f"Unknown controller {ctrl}, skipping")
            continue
        result = run_sequence(cfg, ctrl, args.prewarm, args.run, args.step_interval, args.log_events)
        summarize(result['prewarm_output'], result['measured_output'], ctrl)
        events = result['event_log']
        if args.log_events and events:
            path = f"{args.log_prefix}_{ctrl}.jsonl"
            dirpath = os.path.dirname(path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                for record in events:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Events logged to {path}")
