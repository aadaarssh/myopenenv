from typing import Tuple
from .models import Observation, Action, Reward, Info
from .simulator import APMCSimulator
from .tasks import TASKS

class APMCEnv:
    def __init__(self, task_name: str = "easy_arbitrage"):
        super().__init__()
        self.task_name = task_name
        self.simulator = None
        self.theoretical_max = 0.0
        self.step_count = 0
        self.max_steps = 30 # absolute safety bound

    def reset(self) -> Observation:
        if self.task_name not in TASKS:
            raise ValueError(f"Unknown task {self.task_name}")
        
        state, max_p = TASKS[self.task_name]()
        self.simulator = APMCSimulator(state)
        # Ensure simulator events/lists are strictly cleared on a hard reset
        self.simulator.events.clear()
        self.simulator.predictions.clear()
        self.simulator.known_markets.clear()
        self.simulator.known_logistics.clear()
        
        self.theoretical_max = max_p
        self.step_count = 0
        
        return self.simulator.get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        if not self.simulator:
             raise RuntimeError("Must call reset() before step()")

        self.step_count += 1
        reward_val = self.simulator.process_action(action)
        
        # User defined Done Fix
        done = self.simulator.state.current_day >= self.simulator.state.max_days
        if self.step_count >= self.max_steps:
             done = True
        
        obs = self.simulator.get_observation()
        
        # Robust grading calculation
        profit = self.simulator.state.current_balance
        efficiency = profit / max(1.0, self.theoretical_max)
        penalty_factor = min(1.0, self.simulator.state.current_day / max(1, self.simulator.state.max_days))
        # Reduce score linearly up to 10% based on how many days it took
        grade = max(0.0, min(1.0, efficiency * (1.0 - 0.1 * penalty_factor)))
        
        inventory_waste = self.simulator.state.current_inventory + sum(s.quantity for s in self.simulator.state.in_transit) + sum(v for v in self.simulator.state.inventory_at_markets.values())

        metrics = {
            "events": obs.recent_events.copy(),
            "profit": round(profit, 2),
            "waste": round(inventory_waste, 2),
            "efficiency": round(efficiency, 3)
        }

        info = Info(
            task_id=self.task_name,
            profit_achieved=round(profit, 2),
            theoretical_max=round(self.theoretical_max, 2),
            grade=round(grade, 3),
            metrics=metrics
        )
        
        return obs, Reward(value=reward_val), done, info

    def state(self) -> str:
        if not self.simulator:
            return "{}"
        return self.simulator.state.model_dump_json(indent=2)
