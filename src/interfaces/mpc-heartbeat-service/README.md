# MPC Heartbeat Service

This service reads a heartbeat value on D-Bus and resets `AcPowerSetPoint` to a fallback value when the heartbeat times out.

## First start on Victron GX

Assumption: the service files are available in `/data/mpc-heartbeat-service`.

1. Make scripts executable:

```sh
chmod +x /data/mpc-heartbeat-service/run
chmod +x /data/mpc-heartbeat-service/log/run
```

2. Register service with daemontools and persist the symlink across reboots:

```sh
ln -sf /data/mpc-heartbeat-service /service/mpc-heartbeat-service
ls -la /service/mpc-heartbeat-service
```

The `/service` directory is on a tmpfs and is wiped on every reboot. Add the symlink creation to `/data/rc.local` so it is restored automatically:

```sh
cat >> /data/rc.local << 'EOF'

# MPC Heartbeat Service
sleep 10 && ln -sf /data/mpc-heartbeat-service /service/mpc-heartbeat-service
EOF
chmod +x /data/rc.local
```

The `sleep 10` gives D-Bus and svscan time to start before the service is registered.

3. Start or restart service:

```sh
svc -u /service/mpc-heartbeat-service
# or restart
svc -t /service/mpc-heartbeat-service
```

4. Check status:

```sh
svstat /service/mpc-heartbeat-service
svstat /service/mpc-heartbeat-service/log
```

## Read logs

Live tail:

```sh
tail -f /data/mpc-heartbeat-service/log/main/current
```

Last lines:

```sh
tail -n 100 /data/mpc-heartbeat-service/log/main/current
```

## Notes

- The Python service logs to stderr by default. The service `run` script must keep `exec 2>&1` so the log process receives all messages.
- Timestamps are emitted in UTC in ISO-like format, for example: `2026-03-15T15:55:00Z`.

## Troubleshooting

- If `svc` or `svstat` prints `unable to chdir ... file does not exist`, the service symlink is missing or wrong. The link must be `/service/mpc-heartbeat-service -> /data/mpc-heartbeat-service`. This symlink is lost after every reboot unless the `/data/rc.local` entry is set (see above).
- If startup fails with `dbus.exceptions.NameExistsException: Bus name already exists: com.victronenergy.molu`, another process is already using that D-Bus name. Stop the old process or use a unique service name in `dbus-mpc-heartbeat.py`.
