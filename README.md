# Home_BESS

## Data preparation

Convert utility import/export CSVs into `energy_data_01.csv` for
simulation input by editing the configuration variables in the
`__main__` section of [scripts/generate_energy_data_from_grid_csv.py](scripts/generate_energy_data_from_grid_csv.py).
Then run:

```bash
python scripts/generate_energy_data_from_grid_csv.py
```
