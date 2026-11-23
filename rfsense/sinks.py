"""Outputs for events (stdout, serial to the ESP, udp, log file)."""

from __future__ import annotations

import logging
import socket
from pathlib import Path

from .config import OutputConfig

log = logging.getLogger(__name__)


class StdoutSink:
    def send(self, line: str) -> None:
        print(line, flush=True)

    def close(self) -> None:
        pass


class FileSink:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def send(self, line: str) -> None:
        self.f.write(line + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


class SerialSink:
    """Serial to the ESP. Reopens the port if it gets unplugged or reset."""

    def __init__(self, port: str, baud: int):
        self.port, self.baud = port, baud
        self.ser = None

    def _open(self):
        import serial

        ser = serial.Serial()
        ser.port, ser.baudrate, ser.write_timeout = self.port, self.baud, 1.0
        # keep DTR/RTS low, toggling them resets most ESP dev boards
        ser.dtr = False
        ser.rts = False
        ser.open()
        return ser

    def send(self, line: str) -> None:
        try:
            if self.ser is None:
                self.ser = self._open()
            self.ser.write((line + "\n").encode())
        except Exception as e:
            log.warning("serial %s: %s (event dropped)", self.port, e)
            self.close()

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            finally:
                self.ser = None


class UdpSink:
    """In case we ever talk to the ESP over wifi instead"""

    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, line: str) -> None:
        try:
            self.sock.sendto((line + "\n").encode(), self.addr)
        except OSError as e:
            log.warning("udp %s: %s (event dropped)", self.addr, e)

    def close(self) -> None:
        self.sock.close()


def make_sinks(cfg: OutputConfig) -> list:
    sinks = []
    for name in cfg.sinks:
        if name == "stdout":
            sinks.append(StdoutSink())
        elif name == "file":
            sinks.append(FileSink(cfg.file_path))
        elif name == "serial":
            sinks.append(SerialSink(cfg.serial_port, cfg.serial_baud))
        elif name == "udp":
            sinks.append(UdpSink(cfg.udp_host, cfg.udp_port))
        else:
            raise ValueError(f"unknown sink: {name}")
    return sinks
