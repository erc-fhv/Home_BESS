## 1. Script Setup

- [x] 1.1 Create `scripts/generate_energy_data_from_grid_csv.py` with module docstring and imports
- [x] 1.2 Define constants for default annual PV production (8000 kWh) and timezone (Europe/Vienna)
- [x] 1.3 Add command-line argument parsing for input files and optional annual PV parameter

## 2. CSV Parsing Functions

- [x] 2.1 Implement function to parse semicolon-delimited CSV with decimal commas
- [x] 2.2 Add logic to skip metadata rows until `Beginn der Messreihe` header is found
- [x] 2.3 Extract timestamp pairs (start/end) and energy values in kWh from each row
- [x] 2.4 Write unit tests for CSV parsing edge cases (metadata, decimal commas)

## 3. Timestamp Processing

- [x] 3.1 Implement function to parse DD.MM.YYYY HH:MM format to datetime objects
- [x] 3.2 Localize timestamps to Europe/Vienna timezone with DST error handling
- [x] 3.3 Use interval start time as the index for each record
- [x] 3.4 Write unit tests for timezone parsing and DST edge cases

## 4. Self-Consumption Inference

- [x] 4.1 Calculate data window duration in days from timestamp range
- [x] 4.2 Compute window PV total as annual_pv × (days/365)
- [x] 4.3 Sum total grid export from energy_to_grid CSV
- [x] 4.4 Calculate self-consumed PV as residual (window PV - export)
- [x] 4.5 Add validation to detect and raise error for negative self-consumption
- [x] 4.6 Write unit tests for self-consumption calculation logic

## 5. Self-Consumption Allocation

- [x] 5.1 Sum total grid import from energy_from_grid CSV
- [x] 5.2 Calculate proportional allocation weights per interval (import/total_import)
- [x] 5.3 Distribute total self-consumed PV across intervals using weights
- [x] 5.4 Handle zero-import intervals (allocate zero self-consumption)
- [x] 5.5 Write unit tests for allocation logic

## 6. Output Generation

- [x] 6.1 Build DataFrame with interval start timestamps as index
- [x] 6.2 Compute Production column (export + allocated self-consumption) in Wh
- [x] 6.3 Compute Consumption column (import + allocated self-consumption) in Wh
- [x] 6.4 Write DataFrame to `data/energy_data_01.csv` with timezone-aware index
- [x] 6.5 Write unit tests to validate output schema

## 7. Validation and Summary

- [x] 7.1 Calculate and validate total import, export, production, consumption
- [x] 7.2 Check energy balance (Production = Export + Self-consumption)
- [x] 7.3 Print summary with totals, date range, and annual PV parameter used
- [x] 7.4 Add error handling with clear messages for validation failures

## 8. Documentation and Integration

- [x] 8.1 Add NumPy-style docstrings to all functions with type hints
- [x] 8.2 Create example usage in script docstring showing command-line invocation
- [x] 8.3 Update README or project documentation referencing the new script
- [x] 8.4 Run pre-commit hooks (Ruff linting) and verify compliance

## 10. Script Configuration Adjustment

- [x] 10.1 Replace CLI configuration with main-guard variables in the script

## 9. End-to-End Testing

- [x] 9.1 Test with provided demo files (energy_from_grid_01.csv, energy_to_grid_01.csv)
- [x] 9.2 Verify output file matches energy_data.csv schema
- [x] 9.3 Validate summary output shows correct totals and parameters
- [x] 9.4 Test with different annual PV parameter values
