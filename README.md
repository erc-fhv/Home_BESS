# Home_BESS

## Sytemd service file for running the MPC controller on an server:

Read out the content of the file and save it as /etc/systemd/system/mpc.service:
```
sudo nano /etc/systemd/system/mpc.service
```

```ini
[Unit]
Description=MPC Python Service
After=network.target

[Service]
User=deinuser
WorkingDirectory=/home/molu/repos/Home_BESS/
ExecStart=/home/molu/repos/Home_BESS/.venv/bin/python -u /home/molu/repos/Home_BESS/src/control/mpc.py
Restart=always
RestartSec=60
Environment=PYTHONUNBUFFERED=1
User=molu

[Install]
WantedBy=multi-user.target
```
If content was changed:
```
sudo systemctl daemon-reload
```
Restart the systemd service:
```
sudo systemctl restart mpc.service
```
View the logs of the service:
```
sudo journalctl -u mpc.service -f
```
