from robobopy.Robobo import Robobo

robobo = Robobo("192.168.1.131")
# robobo = Robobo("localhost")
robobo.connect()

while True:
    # robobo.moveWheels(10, 10)
    qr = robobo.readQR()
    if qr != None:
        print(qr.id)
        robobo.sayText("LEYENDO")
    else:
        robobo.sayText("NADA")