import socket

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '192.168.40.152'  # Loopback address for local testing
    port = 8000  # You can choose any available port

    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Server listening on {host}:{port}")

    conn, addr = server_socket.accept()
    print(f"Connection established with {addr}")

    while True:
        command = input("Enter a command to send to the client (or 'exit' to quit): ")

        if command.lower() == 'exit':
            break

        conn.send(command.encode())

        response = conn.recv(1024)
        print(f"Response from client: {response.decode()}")

    conn.close()
    print("Connection closed")


if __name__ == "__main__":
    start_server()