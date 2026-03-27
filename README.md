# Home_BESS

## Sytemd service file for running the MPC controller on an server:

Create the following file:
```
sudo nano /etc/systemd/system/mpc.service
```

Add this content to the file:

```ini
[Unit]
Description=MPC Python Service
After=network.target

[Service]
User=molu
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
View the current status of the service:
```
sudo systemctl status mpc.service
```
