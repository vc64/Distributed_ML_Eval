import re
import socket


HOST = "0.0.0.0"
PORT = 5000


def main() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()

        while True:
            connection, _ = server.accept()
            with connection:
                message = connection.recv(1024).decode().strip()
                match = re.fullmatch(r"ping (\d+)", message)
                if match:
                    worker_id = match.group(1)
                    print(f"i am worker {worker_id}", flush=True)
                    print(f"[{worker_id}] i got pinged", flush=True)
                    connection.sendall(b"ack\n")


if __name__ == "__main__":
    main()
