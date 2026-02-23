## Context

The current simulation uses synthetic PV production and household load data in
`data/energy_data.csv`. We now have utility interval CSVs for grid import and
export, but they only include grid flows, not total site load or total PV
production. The change needs a consistent way to infer self-consumed PV and
produce `energy_data.csv` in the existing schema. Timestamps are provided in
local time and must be interpreted as Europe/Vienna. The data window can be
partial (e.g., two days), so the conversion must scale annual PV assumptions
accordingly. The resulting profiles feed the BESS optimization pipeline and
should remain compatible with other data sources (MQTT SOC, ENTSO-E prices,
Open-Meteo weather).

## Goals / Non-Goals

**Goals:**
- Convert grid import/export CSVs into `energy_data.csv` with `Production` and
  `Consumption` at 15-minute resolution.
- Infer self-consumed PV using an annual PV production assumption (8000 kWh) and
  the observed export totals in the window.
- Preserve timezone-aware timestamps in Europe/Vienna and use interval start
  times as the index.

**Non-Goals:**
- Do not estimate a full year from a short window beyond the stated annual PV
  scaling assumption.
- Do not change the BESS optimization model, forecasting models, or dashboard
  logic.
- Do not introduce new external dependencies beyond existing Python stack.

## Decisions

- **CSV parsing format:** Parse semicolon-delimited files with decimal commas
  and skip the metadata header rows until the `Beginn der Messreihe` header.
  This matches the supplier export format and avoids manual pre-cleaning.
  Alternative: require pre-converted CSVs; rejected to keep the workflow
  repeatable.
- **Timezone handling:** Treat timestamps as Europe/Vienna. Fail on DST
  ambiguity/nonexistent times to surface data issues early. Alternative:
  auto-resolve DST (infer/shift); rejected because the user requested manual
  handling.
- **Self-consumption inference:**
  - Compute window PV total as $PV_{window} = PV_{annual} \times days/365$,
    where $PV_{annual}$ is a configurable parameter (8000 kWh for the demo
    dataset).
  - Compute export total from `energy_to_grid_01.csv`.
  - Self-consumed PV is the residual: $SelfUse = PV_{window} - Export$.
  - Allocate `SelfUse` across intervals proportional to grid import per
    interval (proxy for load shape).
  Alternative: use a synthetic PV profile (pvlib) to allocate self-use; rejected
  to avoid reintroducing synthetic data when measured grid flows are available.
- **Output schema:** Write `Production` and `Consumption` in Wh with the interval
  start timestamp index, matching the existing `energy_data.csv` schema.

## Risks / Trade-offs

- **Negative self-consumption** (Export > $PV_{window}$) -> Mitigation: detect
  and fail with a clear message so the annual PV assumption can be adjusted.
- **Short-window bias** -> Mitigation: document that totals are scaled and only
  represent the provided window, not a modeled seasonal year.
- **Load shape distortion** from proportional allocation -> Mitigation: keep
  allocation strategy configurable for later refinement.

## Migration Plan

- Add a dedicated conversion script or function to generate
  `data/energy_data_01.csv` from the two grid CSVs.
- Configure input paths and annual PV assumptions directly in the script
  `__main__` guard for repeatable runs without a CLI.
- Validate totals (import, export, PV window) and produce a short summary.
- Keep existing synthetic generator available for other scenarios.

## Open Questions

None — design decisions are finalized.
