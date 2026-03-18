"""
Predictive Controller Prototype (memory-enhanced)
-------------------------------------------------

Uses SimpleStateAggregator to build a compact state representation,
maintains a simple latent memory, and learns a linear predictive model to
select ExternalInput actions that keep the ecosystem diverse and stable.
This is still a placeholder for更高级的主动推理/世界模型控制器。
"""
import time
import numpy as np
from dataclasses import dataclass
from typing import List

from engine import core
from .state_aggregator import SimpleStateAggregator, AggregatedState

ACTIONS = ['chem_energy', 'chem_nutrient', 'chem_inhibitor', 'adjust_solar']


def one_hot_action(action: str) -> np.ndarray:
    vec = np.zeros(len(ACTIONS), dtype=np.float32)
    if action in ACTIONS:
        vec[ACTIONS.index(action)] = 1.0
    return vec


@dataclass
class Experience:
    feature: np.ndarray
    action: np.ndarray
    next_state: np.ndarray


class PredictiveController:
    def __init__(self, config: core.Ecology2DConfig, step_interval: float = 0.0,
                 history_capacity: int = 128, hidden_size: int = 16):
        self.system = core.Ecology2DSystem(config)
        self.step_interval = step_interval
        self.aggregator = SimpleStateAggregator(config, grid_size=32, history_length=16)
        self.buffer: List[Experience] = []
        self.max_buffer = history_capacity
        self.state_dim = 10
        self.action_dim = len(ACTIONS)
        self.hidden_size = hidden_size
        self.model = np.zeros((self.state_dim + hidden_size + self.action_dim, self.state_dim), dtype=np.float32)
        self.last_feature = None
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.hidden_state = np.zeros(hidden_size, dtype=np.float32)
        rng = np.random.default_rng(42)
        self.hidden_proj = rng.normal(scale=0.4, size=(self.state_dim, hidden_size)).astype(np.float32)
        self.hidden_mix = rng.normal(scale=0.1, size=(hidden_size, hidden_size)).astype(np.float32)
        self.state_mean = np.zeros(self.state_dim, dtype=np.float32)
        self.state_var = np.ones(self.state_dim, dtype=np.float32)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        self.state_mean = 0.995 * self.state_mean + 0.005 * state
        centered = state - self.state_mean
        self.state_var = 0.995 * self.state_var + 0.005 * (centered ** 2) + 1e-4
        return centered / np.sqrt(self.state_var)

    def _update_hidden(self, norm_state: np.ndarray):
        stim = norm_state @ self.hidden_proj + self.hidden_state @ self.hidden_mix
        self.hidden_state = np.tanh(stim)

    def _feature_vector(self, state_vec: np.ndarray) -> np.ndarray:
        norm_state = self._normalize_state(state_vec)
        self._update_hidden(norm_state)
        return np.concatenate([norm_state, self.hidden_state])

    def _update_model(self, feature: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        if len(self.buffer) >= self.max_buffer:
            self.buffer.pop(0)
        self.buffer.append(Experience(feature.copy(), action.copy(), next_state.copy()))
        xs = []
        ys = []
        for exp in self.buffer:
            xs.append(np.concatenate([exp.feature, exp.action]))
            ys.append(exp.next_state)
        X = np.stack(xs, axis=0)
        Y = np.stack(ys, axis=0)
        try:
            XtX = X.T @ X + np.eye(X.shape[1]) * 1e-3
            XtY = X.T @ Y
            self.model = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            pass

    def _predict(self, feature: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = np.concatenate([feature, action])
        return x @ self.model

    def _score_vector(self, state_vec: np.ndarray) -> float:
        # diversity, avg energy, emergence score, suppress inhibitors
        diversity = state_vec[2]
        avg_energy_scaled = state_vec[1]
        emergence = state_vec[8]
        atp = state_vec[9]
        return 2.0 * diversity + 0.8 * avg_energy_scaled + 1.0 * emergence - 0.3 * abs(atp - 0.3)

    def _choose_action(self, feature: np.ndarray, agg_state: AggregatedState) -> str:
        state_vec = agg_state.vector
        current_score = self._score_vector(state_vec)
        best_action = ACTIONS[0]
        best_gain = -1e9
        for action_name in ACTIONS:
            action_vec = one_hot_action(action_name)
            pred_state = self._predict(feature, action_vec)
            gain = self._score_vector(pred_state) - current_score
            if gain > best_gain:
                best_gain = gain
                best_action = action_name
        return best_action

    def _execute_action(self, action: str, agg_state: AggregatedState, output: core.SystemOutput):
        stats = output.stats
        world = self.system.config.world_size
        grids = agg_state.multi_scale_grids
        hotspot = None
        if grids:
            smallest = min(grids.keys())
            grid = grids[smallest]
            if action == 'chem_nutrient':
                idx = int(np.argmin(grid))
            else:
                idx = int(np.argmax(grid))
            gx, gy = divmod(idx, grid.shape[1])
            scale = world / max(1, grid.shape[0])
            hotspot = (int(gx * scale) % world, int(gy * scale) % world)
        if hotspot is None:
            hotspot = (int(stats['alive_particles'] % world), int((stats['alive_particles'] // 3) % world))
        if action == 'chem_energy':
            params = {'x': hotspot[0], 'y': hotspot[1], 'radius': 10, 'intensity': 1.2}
            event = 'chemical_pulse'
        elif action == 'chem_nutrient':
            params = {'x': hotspot[0], 'y': hotspot[1], 'radius': 12, 'intensity': 1.5, 'chemical_index': 1}
            event = 'chemical_pulse'
        elif action == 'chem_inhibitor':
            params = {'x': hotspot[0], 'y': hotspot[1], 'radius': 12, 'intensity': 1.0, 'chemical_index': 2}
            event = 'chemical_pulse'
        elif action == 'adjust_solar':
            params = {'solar_energy_rate': self.system.config.solar_energy_rate * 1.03}
            event = 'parameter_adjust'
        else:
            return
        self.system.apply_external_input(core.ExternalInput(event, params))

    def run(self, total_steps: int = 2000):
        for step in range(total_steps):
            self.system.step()
            if step % 20 == 0:
                output = self.system.get_system_output(metadata={'source': 'predictive_controller'})
                agg_state = self.aggregator.aggregate(output)
                state_vec = agg_state.vector
                feature = self._feature_vector(state_vec)
                if self.last_feature is not None:
                    self._update_model(self.last_feature, self.last_action, state_vec)
                action_name = self._choose_action(feature, agg_state)
                action_vec = one_hot_action(action_name)
                self._execute_action(action_name, agg_state, output)
                self.last_feature = feature
                self.last_action = action_vec
            if self.step_interval:
                time.sleep(self.step_interval)


if __name__ == "__main__":
    cfg = core.Ecology2DConfig(world_size=128, n_particles=2000)
    controller = PredictiveController(cfg, step_interval=0.0)
    controller.run(total_steps=2000)
