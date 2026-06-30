import pickle
import socket
import struct
from typing import Any


_HEADER = struct.Struct("!Q")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while receiving payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)


def recv_message(sock: socket.socket) -> Any:
    header = _recv_exact(sock, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    return pickle.loads(_recv_exact(sock, size))


def request(host: str, port: int, payload: Any, timeout: float = 600.0) -> Any:
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.settimeout(timeout)
        send_message(sock, payload)
        return recv_message(sock)
