## Why

The project needs a way to derive realistic site load and PV production
inputs from utility import/export CSVs instead of synthetic generation, so
simulations can be grounded in measured data. This is timely because we now
have interval data and want to align optimization inputs with actual grid
flows.

## What Changes

- Add a repeatable conversion from grid import/export CSVs to the
  `energy_data.csv` schema (Production/Consumption at 15-minute resolution).
- Define how to infer self-consumed PV using annual PV production assumptions
  and interval exports.
- Establish handling for timezone parsing and window coverage when CSVs only
  cover partial periods.

## Capabilities

### New Capabilities
- `grid-csv-energy-data`: Convert utility import/export interval CSVs into
  `energy_data.csv` inputs with inferred self-consumption and timezone-aware
  timestamps.

### Modified Capabilities
- 

## Impact

- Data pipeline: new conversion path for `data/energy_data.csv` inputs.
- Simulation/optimization: BESS dispatch uses measured-based profiles instead
  of synthetic ones.
- Interfaces: aligned with grid data alongside existing MQTT (SOC), ENTSO-E
  (prices), and Open-Meteo (weather) inputs used by the control stack.
- Assumptions: requires documented PV annual production and export totals; must
  respect battery constraints when used downstream.
