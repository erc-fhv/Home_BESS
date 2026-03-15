
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
        self.heartbeat_interval_sec = 20
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None

        self.topics = {
            "soc_percent": f"N/{self.portal_id}/battery/512/Soc",
            "keepalive": f"R/{self.portal_id}/keepalive",
            "netload": f"W/{self.portal_id}/settings/0/Settings/CGwacs/AcPowerSetPoint",
            "heartbeat": f"W/{self.portal_id}/molu/0/MpcHeartbeat",
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
            self.start_heartbeat()
        else:
            raise ConnectionError(f"Failed to connect to MQTT broker. Code: {rc}")

    def start_heartbeat(self):
        """
        Start publishing a heartbeat to the MPC heartbeat service every 20 seconds.
        (If not already running.)
        """

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_stop.clear()

        def _heartbeat_loop():
            self.send_heartbeat()
            while not self._heartbeat_stop.wait(self.heartbeat_interval_sec):
                self.send_heartbeat()

        self._heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name="mpc-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def send_heartbeat(self):
        payload = json.dumps({"value": time.monotonic()})
        self.client.publish(self.topics["heartbeat"], payload, retain=False)

    def stop_heartbeat(self):
        """Stop the heartbeat thread."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join()

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
                time.sleep(1)  # Avoid busy waiting

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
