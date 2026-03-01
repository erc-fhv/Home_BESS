from dataclasses import dataclass
import numpy as np

@dataclass
class MpcConfig:
    """Configuration config for the MPC controller."""

    mpc_interval_sec: int = 15 * 60
    max_charge_kw: float = 4.0*3
    max_discharge_kw: float = 4.0*3
    capacity_kwh: float = 5.12*6
    eta: float = np.sqrt(0.95) * 0.96   # 95% battery round-trip efficiency
                                        # multiplied by 96% converter power factor
    eta_charge: float = eta
    eta_discharge: float = eta
    soc_min_percent: float = 10.0
    soc_final_percent: float = 50.0
