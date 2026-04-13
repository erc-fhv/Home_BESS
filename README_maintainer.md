# Home_BESS

## Run the Simulation Web App

Create the following systemd service file:
```
sudo nano /etc/systemd/system/web-app.service
```

Add this content to the file:

```ini
[Unit]
Description=Homebatterie Webservice (Gunicorn)
After=network.target

[Service]
# User and group to run as (change if needed)
User=molu
Group=molu

# Working directory of the app
WorkingDirectory=/home/molu/repos/Home_BESS/src/simulation

# Command to start Gunicorn
ExecStart=/home/molu/repos/Home_BESS/.venv/bin/gunicorn "web_app:create_application()" --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker --workers 1 -b 0.0.0.0:5000

# Restart policy
Restart=always
RestartSec=5

# Environment variables (optional)
# Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target

# If needed improve, see docs: https://docs.gunicorn.org/en/stable/deploy.html
```

Enable and start the service:
```
sudo systemctl daemon-reload
sudo systemctl enable web-app.service
sudo systemctl start web-app.service
```

Restart the service after code changes:
```
sudo systemctl restart web-app.service
```

View the logs:
```
sudo journalctl -u web-app.service -f
```

View the current status:
```
sudo systemctl status web-app.service
```

## Systemd service file for running the MPC controller on a server:

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
Enable auto-start on boot:
```
sudo systemctl enable mpc.service
```
