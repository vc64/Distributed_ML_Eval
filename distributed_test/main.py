import socket
import time


WORKER_HOST = "worker"
WORKER_PORT = 5000
REPLICA_COUNT = 4
RETRY_DELAY_SECONDS = 0.5
MAX_RETRIES = 30


def worker_addresses() -> list[str]:
    addresses = set()
    for _ in range(MAX_RETRIES):
        try:
            results = socket.getaddrinfo(
                WORKER_HOST,
                WORKER_PORT,
                type=socket.SOCK_STREAM,
            )
            addresses.update(result[4][0] for result in results)
            if len(addresses) >= REPLICA_COUNT:
                return sorted(addresses)[:REPLICA_COUNT]
        except socket.gaierror:
            pass
        time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"could not discover {REPLICA_COUNT} worker replicas; "
        f"found {len(addresses)}"
    )


def ping_worker(address: str, worker_id: int) -> None:
    for attempt in range(MAX_RETRIES):
        try:
            with socket.create_connection(
                (address, WORKER_PORT), timeout=2
            ) as connection:
                connection.sendall(f"ping {worker_id}\n".encode())
                connection.recv(1024)
            return
        except OSError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY_SECONDS)


def main() -> None:
    print("i am the main node", flush=True)
    for worker_id, address in enumerate(worker_addresses(), start=1):
        ping_worker(address, worker_id)


if __name__ == "__main__":
    main()
