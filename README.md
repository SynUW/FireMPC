# FireMPC

The historical FireMPC scripts at the top level (`gee_datadownload/`,
`newset_processing/`, `fwi/`, `sampling/`) are the training-dataset workflow.

`operational/` is the near-real-time application, and it is where the active
work is. It takes a range of dates and fills those dates into the monthly
CanFireCast store at `$WILDFIRE_DATA_ROOT/packages/canfirecast/YYYY-MM.h5`,
using sources that can be fetched as close to the date as possible — IFS from
Earth Engine, MODIS through earthaccess.

```bash
conda activate gee_down
python operational/cli.py fill --start 2026-06-01 --end 2026-06-30
```

See [`operational/README.md`](operational/README.md) for how to use it and
[`operational/NOTES.md`](operational/NOTES.md) for what running it taught.
Model inference, maps and evaluation live outside this repository; they read
the store through `canfirecast store publish`.

Note that `operational/` imports from `newset_processing/` and `fwi/`, so
neither can be removed.
