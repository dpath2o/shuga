# shuga SeaIceObservations compatibility patch

This patch fixes imports after splitting the old mixed observation class into
`AF2020Observations` and `NSIDCObservations`.

## Files to copy

```text
shuga/observations/legacy.py
shuga/observations/__init__.py
```

The new `legacy.py` defines a backwards-compatible `SeaIceObservations`
facade. It delegates to the new AF2020 and NSIDC classes but preserves old
method names used by `CICEPlotter`, older scripts, and notebooks.

## Optional tidy patch

```text
shuga/waves/cawcr_import_cleanup.diff
```

`shuga/waves/cawcr.py` imports `SeaIceObservations` and `ObservationSpec` but
does not use either in the current file. Removing those imports reduces future
coupling between waves and observations. This is optional once the compatibility
facade is installed.

## After copying

From the repository root:

```bash
python - <<'PY'
import shuga
from shuga.observations import SeaIceObservations, AF2020Observations, NSIDCObservations
print("shuga import ok")
print(SeaIceObservations, AF2020Observations, NSIDCObservations)
PY
```

Then rerun the AF2020 PBS job.
