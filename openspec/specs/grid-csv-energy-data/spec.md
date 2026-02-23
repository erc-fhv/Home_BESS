## Purpose
Define requirements for converting grid import/export CSV data into
`energy_data.csv` with a timezone-aware 15-minute index.

## Requirements

### Requirement: Parse utility grid CSV format
The system SHALL parse semicolon-delimited grid import and export CSV files
with decimal commas, skipping metadata rows until the `Beginn der Messreihe`
header is found.

#### Scenario: Parse valid grid import CSV
- **WHEN** a CSV file with format `Beginn der Messreihe;Ende der Messreihe;Messwert in kWh` is provided
- **THEN** the system extracts timestamp pairs and energy values in kWh

#### Scenario: Handle decimal comma format
- **WHEN** energy values use comma as decimal separator (e.g., `0,130`)
- **THEN** the system correctly parses them as floating-point numbers

#### Scenario: Skip metadata header rows
- **WHEN** CSV contains metadata rows before the interval data header
- **THEN** the system skips rows until `Beginn der Messreihe` is found

### Requirement: Parse timestamps with timezone awareness
The system SHALL interpret timestamps as Europe/Vienna timezone and use
interval start times as the index.

#### Scenario: Parse local timestamps
- **WHEN** timestamps are in format `DD.MM.YYYY HH:MM` without timezone info
- **THEN** the system interprets them as Europe/Vienna local time

#### Scenario: Use interval start as index
- **WHEN** an interval spans `01.01.2026 00:00` to `01.01.2026 00:15`
- **THEN** the index timestamp SHALL be `01.01.2026 00:00`

#### Scenario: Fail on DST ambiguity
- **WHEN** a timestamp falls in a DST overlap or gap period
- **THEN** the system SHALL raise an error with a clear message

### Requirement: Infer self-consumed PV energy
The system SHALL compute self-consumed PV as the residual between scaled annual
PV production and observed grid export totals within the data window.

#### Scenario: Compute window PV total from annual assumption
- **WHEN** annual PV production is 8000 kWh and the window covers 2 days
- **THEN** window PV total SHALL be 8000 x (2/365) approx 43.8 kWh

#### Scenario: Calculate self-consumption residual
- **WHEN** window PV total is 43.8 kWh and export total is 10.0 kWh
- **THEN** self-consumed PV SHALL be 33.8 kWh

#### Scenario: Detect negative self-consumption
- **WHEN** export total exceeds the window PV total
- **THEN** the system SHALL raise an error suggesting adjustment of annual PV assumption

### Requirement: Allocate self-consumed PV across intervals
The system SHALL distribute total self-consumed PV across 15-minute intervals
proportional to grid import per interval.

#### Scenario: Allocate proportional to import
- **WHEN** self-consumed PV is 10.0 kWh and interval A has double the import of interval B
- **THEN** interval A SHALL receive twice the allocated self-consumption of interval B

#### Scenario: Handle zero import intervals
- **WHEN** an interval has zero grid import
- **THEN** allocated self-consumption for that interval SHALL be zero

### Requirement: Generate energy_data.csv output
The system SHALL produce a CSV file with columns `Production` and `Consumption`
in Wh at 15-minute resolution, indexed by Europe/Vienna timestamps.

#### Scenario: Compute Production per interval
- **WHEN** an interval has 100 Wh export and 50 Wh allocated self-consumption
- **THEN** Production SHALL be 150 Wh

#### Scenario: Compute Consumption per interval
- **WHEN** an interval has 200 Wh import and 50 Wh allocated self-consumption
- **THEN** Consumption SHALL be 250 Wh

#### Scenario: Write output with correct schema
- **WHEN** generating the output CSV
- **THEN** the file SHALL have columns `Production` and `Consumption` in Wh and a timezone-aware datetime index

### Requirement: Validate and summarize conversion
The system SHALL validate input totals, output totals, and produce a summary
of the conversion with key statistics.

#### Scenario: Display totals summary
- **WHEN** conversion completes successfully
- **THEN** the system SHALL display total import, export, window PV, self-consumption, and date range

#### Scenario: Validate energy balance
- **WHEN** computing totals
- **THEN** total Production SHALL equal total export plus total self-consumption

#### Scenario: Report conversion parameters
- **WHEN** conversion completes
- **THEN** the system SHALL display the annual PV production parameter used

### Requirement: Support configurable parameters
The system SHALL accept annual PV production as a configurable parameter per
dataset.

#### Scenario: Override default annual PV production
- **WHEN** user specifies annual PV production as 7000 kWh
- **THEN** the system SHALL use 7000 kWh for window PV total calculation

#### Scenario: Use default for demo dataset
- **WHEN** no annual PV production is specified for demo dataset
- **THEN** the system SHALL default to 8000 kWh
