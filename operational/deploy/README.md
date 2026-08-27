# Daily operational run under systemd

The service works from `/home/quinn/FireMPC` and invokes the `gee_down` conda
interpreter on the runner script directly -- there is nothing to install and no
`PYTHONPATH` to set:

```bash
/path/to/conda/envs/gee_down/bin/python operational/daily.py --target today
```

The daily runner advances the certified state ledger, creates and validates the
CanFireCast package, and writes logs/manifests below `WILDFIRE_DATA_ROOT`. An
exclusive lock prevents overlapping runs; existing retry, timeout, and retention
policies apply.

## Install

1. Update the interpreter path in `wildfire-daily.service` for the deployment.
2. Install and edit the machine-local environment file:

   ```bash
   cd /home/quinn/FireMPC/operational
   sudo install -m 0600 config/operational.env.example /etc/wildfire-daily.env
   sudo editor /etc/wildfire-daily.env
   ```

3. Install the unit and timer:

   ```bash
   sudo cp deploy/wildfire-daily.service deploy/wildfire-daily.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now wildfire-daily.timer
   ```

Inspect it with:

```bash
systemctl list-timers wildfire-daily.timer
systemctl start wildfire-daily.service
journalctl -u wildfire-daily.service -n 100
```

## Required deployment assets

- mounted writable `WILDFIRE_DATA_ROOT`;
- Earthdata and Earth Engine credentials for the service user;
- GDAL command-line tools;
- an updated and verified `gee_down` conda environment;
- a verified immutable CanFireCast release;
- a certified, gap-free state ledger through the required `D-11` date.

Keep data, credentials, source static rasters, state, H5 packages, checkpoints,
and predictions outside Git.
