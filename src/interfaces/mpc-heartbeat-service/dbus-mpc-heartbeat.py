#!/usr/bin/env python3

import sys
import time
import logging
import dbus
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)sZ %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
log = logging.getLogger("DbusMoluHeartbeatService")

if '/opt/victronenergy/dbus-systemcalc-py/ext/velib_python' not in sys.path:
    sys.path.insert(1, '/opt/victronenergy/dbus-systemcalc-py/ext/velib_python')

from vedbus import VeDbusService

TIMEOUT_SEC=120      # Sekunden bis Fallback bei fehlendem Heartbeat
FALLBACK_WATT=50
CHECK_INTERVAL_SEC=5

class MpcHeartbeatService:
    def __init__(self):

        log.info("Starting MPC Heartbeat Service")

        self.last_heartbeat = time.time()

        self.service = VeDbusService("com.victronenergy.molu", register=False)
        self.service.add_path('/DeviceInstance', 0)
        self.service.add_path('/Connected', 1)
        self.service.add_path('/ProductName', 'MPC Heartbeat')
        self.service.add_path('/FirmwareVersion', '1.0')
        self.service.add_path('/MpcHeartbeat', 0, writeable=True, onchangecallback=self._on_change)
        self.service.register()

        self.bus = dbus.SystemBus()

        GLib.timeout_add(CHECK_INTERVAL_SEC * 1000, self.check_watchdog)

    def _on_change(self, path, value):
        # log.info("MpcHeartbeat updated: %s", value)
        self.last_heartbeat = time.time()
        return True

    def check_watchdog(self):
        if time.time() - self.last_heartbeat > TIMEOUT_SEC:
            log.warning("Heartbeat timeout -> resetting AcPowerSetPoint")

            try:
                obj = self.bus.get_object(
                    "com.victronenergy.settings",
                    "/Settings/CGwacs/AcPowerSetPoint"
                )

                iface = dbus.Interface(obj, 'com.victronenergy.BusItem')
                iface.SetValue(FALLBACK_WATT)
                self.last_heartbeat = time.time()

            except Exception as e:
                log.error("Failed to set AcPowerSetPoint: %s", e)

        return True

def main():
    DBusGMainLoop(set_as_default=True)
    MpcHeartbeatService()
    GLib.MainLoop().run()

if __name__ == "__main__":
    main()