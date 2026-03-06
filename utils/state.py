from dataclasses import dataclass
import mujoco
import numpy as np

@dataclass
class MujocoState:
    data: mujoco.MjData
    obs: np.ndarray
    reward: float
    done: bool
    metrics: dict
    info: dict
    
    def replace(self, **kwargs):
        """Replace fields in the MujocoState."""
        return MujocoState(
            data=kwargs.get('data', self.data),
            obs=kwargs.get('obs', self.obs),
            reward=kwargs.get('reward', self.reward),
            done=kwargs.get('done', self.done),
            metrics=kwargs.get('metrics', self.metrics),
            info=kwargs.get('info', self.info)
        )
