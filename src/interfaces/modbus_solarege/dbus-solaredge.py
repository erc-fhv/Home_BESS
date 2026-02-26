#!/usr/bin/env python

from dbus.mainloop.glib import DBusGMainLoop

try:
    import gobject  # Python 2.x
except ImportError:
    from gi.repository import GLib as gobject  # Python 3.x

import ctypes
import dbus
import logging
import math
import os
import platform
import sys
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.payload import BinaryPayloadDecoder

log = logging.getLogger("DbusSolarEdge")

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '/opt/victronenergy/dbus-modem'))
from vedbus import VeDbusService

# ----------------------------------------------------------------
VERSION = "0.2"
SERVER_PORT = 1502
SERVER_HOST = "10.0.0.9"
UNIT = 1
# ----------------------------------------------------------------
CONNECTION = "ModbusTCP " + SERVER_HOST + ":" + str(SERVER_PORT) + ", UNIT " + str(UNIT)

maxPower = 0
PV_SERVICE_KEY = 'pvinverter.pv0'

DBusGMainLoop(set_as_default=True)

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

log.info('Startup, trying connection to Modbus-Server: ' + CONNECTION)

modbusClient = ModbusClient(SERVER_HOST, port=SERVER_PORT, retry_on_empty=True)
modbusClient.auto_open = True

if not modbusClient.is_socket_open():
    if not modbusClient.connect():
        log.warning("unable to connect to " + SERVER_HOST + ":" + str(SERVER_PORT) + ", service will retry in update loop")
    else:
        log.info('Connected to Modbus Server.')
else:
    log.info('Connected to Modbus Server.')

if maxPower != 0:
    log.info('Inverter maxPower manually set to %s W' % maxPower)


def _get_string(regs):
    numbers = []
    for x in regs:
        if ((x >> 8) & 0xFF) != 0:
            numbers.append((x >> 8) & 0xFF)
        if ((x >> 0) & 0xFF) != 0:
            numbers.append((x >> 0) & 0xFF)
    return "".join(map(chr, numbers))


def _get_signed_short(regs):
    return ctypes.c_short(regs).value


def _get_scale_factor(regs):
    return 10 ** _get_signed_short(regs)


def _get_victron_pv_state(state):
    if state == 1:
        return 0
    elif state == 3:
        return 1
    elif state == 4:
        return 11
    elif state == 5:
        return 12
    elif state == 7:
        return 10
    else:
        return 8


class SystemBus(dbus.bus.BusConnection):
    def __new__(cls):
        return dbus.bus.BusConnection.__new__(cls, dbus.bus.BusConnection.TYPE_SYSTEM)


class SessionBus(dbus.bus.BusConnection):
    def __new__(cls):
        return dbus.bus.BusConnection.__new__(cls, dbus.bus.BusConnection.TYPE_SESSION)


def dbusconnection():
    return SessionBus() if 'DBUS_SESSION_BUS_ADDRESS' in os.environ else SystemBus()


def _update():
    global maxPower
    try:
        if not modbusClient.is_socket_open() and not modbusClient.connect():
            log.error('unable to reconnect to %s:%s', SERVER_HOST, SERVER_PORT)
            return True

        regs = modbusClient.read_holding_registers(40071, 38, unit=UNIT)

        if regs.isError():
            log.error('regs.isError: %s', regs)
            return True
        else:
            sf = _get_scale_factor(regs.registers[4])
            dbusservice[PV_SERVICE_KEY]['/Ac/L1/Current'] = round(regs.registers[1] * sf, 2)
            dbusservice[PV_SERVICE_KEY]['/Ac/L2/Current'] = round(regs.registers[2] * sf, 2)
            dbusservice[PV_SERVICE_KEY]['/Ac/L3/Current'] = round(regs.registers[3] * sf, 2)

            sf = _get_scale_factor(regs.registers[11])
            dbusservice[PV_SERVICE_KEY]['/Ac/L1/Voltage'] = round(regs.registers[8] * sf, 2)
            dbusservice[PV_SERVICE_KEY]['/Ac/L2/Voltage'] = round(regs.registers[9] * sf, 2)
            dbusservice[PV_SERVICE_KEY]['/Ac/L3/Voltage'] = round(regs.registers[10] * sf, 2)

            sf = _get_scale_factor(regs.registers[13])
            acpower = _get_signed_short(regs.registers[12]) * sf
            dbusservice[PV_SERVICE_KEY]['/Ac/Power'] = acpower
            dbusservice[PV_SERVICE_KEY]['/Ac/L1/Power'] = round(_get_signed_short(regs.registers[12]) * sf / 3, 2)
            dbusservice[PV_SERVICE_KEY]['/Ac/L2/Power'] = round(_get_signed_short(regs.registers[12]) * sf / 3, 2)
            dbusservice[PV_SERVICE_KEY]['/Ac/L3/Power'] = round(_get_signed_short(regs.registers[12]) * sf / 3, 2)

            sf = _get_scale_factor(regs.registers[24])
            dbusservice[PV_SERVICE_KEY]['/Ac/Energy/Forward'] = float((regs.registers[22] << 16) + regs.registers[23]) * sf / 1000
            dbusservice[PV_SERVICE_KEY]['/Ac/L1/Energy/Forward'] = float((regs.registers[22] << 16) + regs.registers[23]) * sf / 3 / 1000
            dbusservice[PV_SERVICE_KEY]['/Ac/L2/Energy/Forward'] = float((regs.registers[22] << 16) + regs.registers[23]) * sf / 3 / 1000
            dbusservice[PV_SERVICE_KEY]['/Ac/L3/Energy/Forward'] = float((regs.registers[22] << 16) + regs.registers[23]) * sf / 3 / 1000

            dbusservice[PV_SERVICE_KEY]['/StatusCode'] = _get_victron_pv_state(regs.registers[36])
            dbusservice[PV_SERVICE_KEY]['/ErrorCode'] = regs.registers[37]

            regs = modbusClient.read_holding_registers(address=0xF142, count=2, unit=UNIT)
            if regs.isError():
                log.error('regs.isError: %s', regs)
                return True
            decoder = BinaryPayloadDecoder.fromRegisters(regs.registers, byteorder=Endian.Big, wordorder=Endian.Little)
            dbusservice[PV_SERVICE_KEY]['/Ac/AdvancedPwrControlEn'] = decoder.decode_32bit_int()

            regs = modbusClient.read_holding_registers(address=0xF304, count=2, unit=UNIT)
            if regs.isError():
                log.error('regs.isError: %s', regs)
                return True
            decoder = BinaryPayloadDecoder.fromRegisters(regs.registers, byteorder=Endian.Big, wordorder=Endian.Little)
            decodedMaxPower = decoder.decode_32bit_float()
            if maxPower != decodedMaxPower:
                log.info('Inverter maxPower received: %s W' % decodedMaxPower)
            maxPower = decodedMaxPower
            dbusservice[PV_SERVICE_KEY]['/Ac/MaxPower'] = decodedMaxPower

            regs = modbusClient.read_holding_registers(address=0xF001, count=1, unit=UNIT)
            if regs.isError():
                log.error('regs.isError: %s', regs)
                return True
            decoder = BinaryPayloadDecoder.fromRegisters(regs.registers, byteorder=Endian.Big, wordorder=Endian.Little)
            powerlimitRel = decoder.decode_16bit_uint()
            dbusservice[PV_SERVICE_KEY]['/Ac/ActivePowerLimit'] = powerlimitRel
            powerlimit = int(maxPower * powerlimitRel / 100)
            dbusservice[PV_SERVICE_KEY]['/Ac/PowerLimit'] = powerlimit

    except Exception:
        log.error('exception in _update.', exc_info=True)
        return True

    return True


def _handleAdvancedPwrControlEn(path, value):
    log.info("_handleAdvancedPwrControlEn(): someone else updated %s to %s" % (path, value))

    try:
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
        builder.add_32bit_int(value)
        result = modbusClient.write_registers(address=0xF142, values=builder.to_registers(), unit=UNIT)

        if result.isError():
            log.error('result.isError(): ' + str(result))

    except Exception as e:
        log.error('Exception in _handleAdvancedPwrControlEn(): %s' % e)
        return False

    return True


def _handlePowerLimit(path, value):
    if maxPower == 0:
        log.warning('_handlePowerLimit(): maxPower is unknown, unable to set power limit')
        return False

    relVal = min(100, max(0, int(math.ceil(100 * value / maxPower))))

    if (relVal < 0) or (relVal > 100):
        log.warning('_handlePowerLimit(): value %s (%s) out of scope (0 <= value <= 100' % (relVal, value))
        return False

    try:
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
        builder.add_32bit_int(1)
        result = modbusClient.write_registers(address=0xF142, values=builder.to_registers(), unit=UNIT)
        if result.isError():
            log.error('result.isError(): ' + str(result))

        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
        builder.add_16bit_uint(relVal)
        result = modbusClient.write_registers(address=0xF001, values=builder.to_registers(), unit=UNIT)
        if result.isError():
            log.error('result.isError(): ' + str(result))

        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
        builder.add_16bit_int(1)
        result = modbusClient.write_registers(address=0xF100, values=builder.to_registers(), unit=UNIT)
        if result.isError():
            log.error('result.isError(): ' + str(result))

    except Exception as e:
        log.error('Exception in _handlePowerLimit(): %s' % e)
        return False

    return True


def _handleActivePowerLimit(path, value):
    log.info("_handleActivePowerLimit(): someone else updated %s to %s" % (path, value))
    if (value < 0) or (value > 100):
        log.warning("_handleActivePowerLimit(): value out of scope (0 <= value <= 100")
        return False

    try:
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
        builder.add_16bit_uint(value)
        result = modbusClient.write_registers(address=0xF001, values=builder.to_registers(), unit=UNIT)
        if result.isError():
            log.error('result.isError(): ' + str(result))

        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
        builder.add_16bit_int(1)
        result = modbusClient.write_registers(address=0xF100, values=builder.to_registers(), unit=UNIT)
        if result.isError():
            log.error('result.isError(): ' + str(result))

    except Exception as e:
        log.error('Exception in _handleActivePowerLimit(): %s' % e)
        return False

    return True


def new_pv_service(base, service_type, physical, device_id, instance):
    self = VeDbusService("{}.{}.{}_id{:02d}".format(base, service_type, physical, device_id), dbusconnection(), register=False)

    self.add_path('/Mgmt/ProcessName', __file__)
    self.add_path('/Mgmt/ProcessVersion', 'Unkown version, and running on Python ' + platform.python_version())
    self.add_path('/Connected', 1)
    self.add_path('/HardwareVersion', 0)

    _kwh = lambda p, v: (str(round(v, 3)) + 'kWh')
    _a = lambda p, v: (str(round(v, 2)) + 'A')
    _w = lambda p, v: (str(round(v, 2)) + 'W')
    _v = lambda p, v: (str(round(v, 2)) + 'V')
    _p = lambda p, v: (str(min(100, max(0, int(math.ceil(100 * v / maxPower))))) + '%') if maxPower > 0 else 'null'

    firmware = 'unknown'
    serial = 'unknown'
    product_name = 'SolarEdge PV Inverter'

    if modbusClient.is_socket_open():
        regs = modbusClient.read_holding_registers(40004, 56, unit=UNIT)
        if regs.isError():
            log.warning('initial metadata read failed: %s', regs)
        else:
            firmware = _get_string(regs.registers[32:47])
            serial = _get_string(regs.registers[48:55])
            product_name = _get_string(regs.registers[0:15]) + " " + _get_string(regs.registers[16:31])

    self.add_path('/DeviceInstance', instance)
    self.add_path('/FirmwareVersion', firmware)
    self.add_path('/DataManagerVersion', VERSION)
    self.add_path('/Serial', serial)
    self.add_path('/Mgmt/Connection', CONNECTION)
    self.add_path('/ProductId', 41284)
    self.add_path('/ProductName', product_name)
    self.add_path('/Ac/Energy/Forward', None, gettextcallback=_kwh)
    self.add_path('/Ac/Power', None, gettextcallback=_w)
    self.add_path('/Ac/L1/Current', None, gettextcallback=_a)
    self.add_path('/Ac/L2/Current', None, gettextcallback=_a)
    self.add_path('/Ac/L3/Current', None, gettextcallback=_a)
    self.add_path('/Ac/L1/Energy/Forward', None, gettextcallback=_kwh)
    self.add_path('/Ac/L2/Energy/Forward', None, gettextcallback=_kwh)
    self.add_path('/Ac/L3/Energy/Forward', None, gettextcallback=_kwh)
    self.add_path('/Ac/L1/Power', None, gettextcallback=_w)
    self.add_path('/Ac/L2/Power', None, gettextcallback=_w)
    self.add_path('/Ac/L3/Power', None, gettextcallback=_w)
    self.add_path('/Ac/L1/Voltage', None, gettextcallback=_v)
    self.add_path('/Ac/L2/Voltage', None, gettextcallback=_v)
    self.add_path('/Ac/L3/Voltage', None, gettextcallback=_v)
    self.add_path('/Ac/MaxPower', None, gettextcallback=_w)
    self.add_path('/ErrorCode', None)
    self.add_path('/Position', 1)
    self.add_path('/StatusCode', None)
    self.add_path('/Ac/PowerLimit', value=None, description='ESS zero feed-in power limit in W', writeable=True, onchangecallback=_handlePowerLimit, gettextcallback=_w)
    self.add_path('/Ac/AdvancedPwrControlEn', value=None, description='Enable SolarEdge power limitation', writeable=True, onchangecallback=_handleAdvancedPwrControlEn)
    self.add_path('/Ac/ActivePowerLimit', value=None, description='SolarEdge active power limit in %', writeable=True, onchangecallback=_handleActivePowerLimit, gettextcallback=_p)

    self.register()

    return self


dbusservice = {}

base = 'com.victronenergy'
dbusservice[PV_SERVICE_KEY] = new_pv_service(base, 'pvinverter.pv0', 'pvinverter', 0, 20)

gobject.timeout_add(5000, _update)

log.info('Connected to dbus, and switching over to GLib.MainLoop() (= event based)')

mainloop = gobject.MainLoop()
mainloop.run()
