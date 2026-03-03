
from pathlib import Path
import json
import time
import threading
import paho.mqtt.client as mqtt
import ssl

class Victron_Mqtt_Reader:
    def __init__(self):

        # Define MQTT connection parameters
        pw_file = Path(__file__).parent.parent.parent.parent / ".json"
        with open(pw_file, encoding="utf-8") as f:
            my_file = json.load(f)
        self.portal_id = my_file["PORTAL_ID"]
        self.username = my_file["USER"]
        self.pw = my_file["VR_PW"]
        self.broker = "mqtt.victronenergy.com"
        self.port = 8883
        self.latest_packets = {}
        self.watchdog_interval_sec = 20
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = None

        self.topics = {
            "soc_percent": f"N/{self.portal_id}/battery/512/Soc",
            "keepalive": f"R/{self.portal_id}/keepalive",
            "watchdog": f"R/{self.portal_id}/battery/512/CustomName",
            "netload": f"W/{self.portal_id}/settings/0/Settings/CGwacs/AcPowerSetPoint",

            # Using CustomName topic for watchdog, as it unused in this setup.
            "watchdog": f"W/{self.portal_id}/battery/512/CustomName",
        }

        self.connect()

    def connect(self):
        """Connect to the MQTT broker and start the loop."""

        self.client = mqtt.Client(
            client_id="milp_client",
            protocol=mqtt.MQTTv311,
        )
        self.client.username_pw_set(self.username, self.pw)
        cert_file = Path(__file__).parent / "venus-ca.crt"
        self.client.tls_set(
            ca_certs= str(cert_file),  # Victron CA certificate
            cert_reqs=ssl.CERT_REQUIRED
        )
        self.client.tls_insecure_set(False)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.broker, self.port, 60)

        # Start the loop in a separate thread
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.start_watchdog()
        else:
            raise ConnectionError(f"Failed to connect to MQTT broker. Code: {rc}")

    def start_watchdog(self):
        """Start publishing a watchdog packet every 20 seconds."""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return

        self._watchdog_stop.clear()

        def _watchdog_loop():
            self.send_watchdog()
            while not self._watchdog_stop.wait(self.watchdog_interval_sec):
                self.send_watchdog()

        self._watchdog_thread = threading.Thread(
            target=_watchdog_loop,
            name="victron-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def send_watchdog(self):
        payload = json.dumps({"value": str(int(time.time()))})
        self.client.publish(self.topics["watchdog"], payload, retain=True)

    def on_message(self, client, userdata, msg):
        payload_text = msg.payload.decode()

        try:
            self.latest_packets[msg.topic] = json.loads(payload_text)
        except json.JSONDecodeError as e:
            raise ValueError((f"Failed to decode JSON payload for topic {msg.topic}: {payload_text}"
                f"Error: {e}")) from e

        self.latest_packets[msg.topic] = self.latest_packets[msg.topic].get("value")

    def get_latest_value(self, value_type:str, timeout_sec:int=10):
        """
        Return the latest value to a given topic, e.g. "soc_percent".
        """

        if value_type not in self.topics:
            raise ValueError((f"Invalid value type: {value_type}. Valid types are: "
                             f"{list(self.topics.keys())}"))

        topic = self.topics[value_type]
        self.client.subscribe(topic)
        self.send_keepalive(self.client)
        self.latest_packets = {}

        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout_sec:
            if topic in self.latest_packets:
                return self.latest_packets[topic]
            else:
                self.send_keepalive(self.client)
                time.sleep(0.1)  # Avoid busy waiting

        assert False, f"Timeout: No value received for topic {topic} within {timeout_sec} seconds."

    def send_keepalive(self, client):
        """Victron MQTT requires a keepalive message to trigger the data flow."""
        payload = json.dumps({"keepalive-options": ["full"]})
        client.publish(self.topics["keepalive"], payload)

    def set_netload(
        self,
        netload_kw:float | None = None,
        verbose:bool=True,
        ):
        """Set the net load (in kW) on the Victron system via MQTT."""

        if netload_kw is None:
            # Set to default = 50W
            netload_kw = 0.05

        if verbose:
            print(f"Setting net load to {netload_kw:.2f} kW (forecast: {netload_kw:.2f} kW)")

        netload_w = netload_kw * 1000.0  # Convert from kW to W

        payload = json.dumps({"value": netload_w})
        self.client.publish(
            self.topics["netload"],
            payload,
            retain=False, # False, in order to detect connection losses on the edge device.
            )

if __name__ == "__main__":
    mqtt_reader = Victron_Mqtt_Reader()
    latest_soc_percent = mqtt_reader.get_latest_value("soc_percent")
    print(f"Latest SOC Percent: {latest_soc_percent}")
