import json
import ssl
import time
from pathlib import Path

import paho.mqtt.client as mqtt


class Victron_Mqtt_Reader:
    def __init__(self):

        # Define MQTT connection parameters
        pw_file = Path.cwd().parent.parent / ".json"
        with open(pw_file, encoding="utf-8") as f:
            my_file = json.load(f)
        self.portal_id = my_file["PORTAL_ID"]
        self.username = my_file["USER"]
        self.pw = my_file["VR_PW"]
        self.broker = "mqtt.victronenergy.com"
        self.port = 8883
        self.keepalive_topic = f"R/{self.portal_id}/keepalive"
        self.latest_packets = {}

        self.topics = {
            "soc": f"N/{self.portal_id}/battery/512/Soc",
        }

        self.connect()

    def connect(self):
        """Connect to the MQTT broker and start the loop."""
        self.client = mqtt.Client(
            client_id="milp_client",
            protocol=mqtt.MQTTv311,
        )
        self.client.username_pw_set(self.username, self.pw)
        self.client.tls_set(
            ca_certs="../data/venus-ca.crt",  # Victron CA certificate
            cert_reqs=ssl.CERT_REQUIRED,
        )
        self.client.tls_insecure_set(False)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.broker, self.port, 60)

        # Start the loop in a separate thread
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            pass
        else:
            raise ConnectionError(f"Failed to connect to MQTT broker. Code: {rc}")

    def on_message(self, client, userdata, msg):
        payload_text = msg.payload.decode()

        try:
            self.latest_packets[msg.topic] = json.loads(payload_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON payload for topic {msg.topic}: {payload_text}Error: {e}") from e

        self.latest_packets[msg.topic] = self.latest_packets[msg.topic].get("value")

    def get_latest_value(self, value_type: str, timeout_sec: int = 10):
        """Return the latest value to a given topic, e.g. "soc"."""
        if value_type not in self.topics:
            raise ValueError(f"Invalid value type: {value_type}. Valid types are: {list(self.topics.keys())}")

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
        client.publish(self.keepalive_topic, payload)


if __name__ == "__main__":
    mqtt_reader = Victron_Mqtt_Reader()
    latest_soc = mqtt_reader.get_latest_value("soc")
    print(f"Latest SOC: {latest_soc}")
