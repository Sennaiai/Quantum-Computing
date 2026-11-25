"""Network denial used by controlled offline verification, not a security sandbox."""
from __future__ import annotations

import socket


def install():
    def denied(*args, **kwargs):
        raise RuntimeError("Network access is disabled during offline ADMET verification")

    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
