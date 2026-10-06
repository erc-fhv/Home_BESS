# Overview

This repository contains two complementary projects:

1. **Web-based simulation tool** – An easy-to-use simulation for evaluating whether investing in a home battery energy storage system (BESS) is economically beneficial for a household with photovoltaic (PV) generation. Try the online dashboard at **https://homebattery.labs.fhv.at/**.

2. **Real-world Model Predictive Control (MPC)** – An MPC framework for residential battery energy storage systems that optimizes battery operation using mixed-integer linear programming (MILP). The controller leverages real-time electricity prices and weather-based net load forecasting to minimize operating costs.

# Related Paper and Citation

A detailed description of the real-world MPC implementation and an extensive evaluation can be found in our paper:

> [energy.acm.org/eir/real-world-model-predictive-control-for-home-battery-systems-towards-closing-the-simulation-to-reality-gap](https://energy.acm.org/eir/real-world-model-predictive-control-for-home-battery-systems-towards-closing-the-simulation-to-reality-gap/)

If you use this repository in your research, please cite:

```bibtex
@article{Moosbrugger2026,
  author  = {Lukas Moosbrugger and Valentin Seiler and Philipp Wohlgenannt and Sashko Ristov and Peter Kepplinger},
  title   = {Real-World Model Predictive Control for Home Battery Systems: Towards Closing the Simulation-to-Reality Gap},
  journal = {ACM SIGEnergy Energy Informatics Review},
  volume  = {6},
  number  = {3},
  month   = sep,
  year    = {2026},
  url     = {https://energy.acm.org/eir/real-world-model-predictive-control-for-home-battery-systems-towards-closing-the-simulation-to-reality-gap/}
}
````

## Acknowledgments

<a href="https://projekte.ffg.at/projekt/4597880">
  <img src="FFG_Logo.png" alt="FFG Logo" width="180" align="right" style="margin-right:16px; margin-bottom:8px;">
</a>

This work was financially supported by the Austrian Research Promotion Agency (FFG) through the **Hub4FlECs** project (COIN FFG 898053). We gratefully acknowledge the FFG for funding the development of the software presented in this repository.

Project page: https://projekte.ffg.at/projekt/4597880

<br clear="left">

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
