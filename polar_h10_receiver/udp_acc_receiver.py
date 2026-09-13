import socket
import math

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening UDP on port {UDP_PORT}...")
print()
print("Polar -> KFall mapping")
print("KFall X =  Polar Y")
print("KFall Y = -Polar X")
print("KFall Z = -Polar Z   # tạm thời, cần verify supine với KFall")
print()

while True:
    data, addr = sock.recvfrom(1024)

    text = data.decode("utf-8").strip()

    try:
        timestamp, x, y, z = text.split(",")

        timestamp = int(timestamp)

        # =====================================
        # Polar H10 RAW: mg -> g
        # =====================================
        px = float(x) / 1000.0
        py = float(y) / 1000.0
        pz = float(z) / 1000.0

        # =====================================
        # Polar coordinate -> KFall coordinate
        # =====================================

        kx = py
        ky = -px
        kz = -pz

        magnitude = math.sqrt(
            kx * kx +
            ky * ky +
            kz * kz
        )

        print(
            f"Polar: "
            f"X={px:+.3f} "
            f"Y={py:+.3f} "
            f"Z={pz:+.3f}"
            f"  ->  "
            f"KFall: "
            f"X={kx:+.3f} "
            f"Y={ky:+.3f} "
            f"Z={kz:+.3f} "
            f"|A|={magnitude:.3f}g"
        )

    except Exception as e:
        print("Parse error:", text, e)