from robobopy.Robobo import Robobo
from EvitarColisiones import EvitarColisiones
from SeguirCarril import SeguirCarril
from LeerQR import LeerQR
import time


def main():
    robobo = Robobo("localhost")
    robobo.connect()

    # diccionario que se pasará a los comportamientos para que lo activen cuando se finalice la misión
    params = {"stop": False}

    # creamos de los comportamientos
    seguir_carril_comportamiento = SeguirCarril(robobo, [], params)
    leer_qr_comportamiento = LeerQR(robobo, [seguir_carril_comportamiento], params)
    evitar_colisiones_comportamiento = EvitarColisiones(robobo, [seguir_carril_comportamiento, leer_qr_comportamiento], params)

    # lista con todos los comportamientos
    threads = [seguir_carril_comportamiento, leer_qr_comportamiento, evitar_colisiones_comportamiento]

    # iniciamos todos los comportamientos
    seguir_carril_comportamiento.start()
    leer_qr_comportamiento.start()
    evitar_colisiones_comportamiento.start()

    # el hilo princimal se mantiene en espera hasta que algún comportamiento marca el objetivo como terminado
    while not params["stop"]:
        time.sleep(0.1)

    # espera a que terminen todos los hilos
    for thread in threads:
        thread.join()

    robobo.disconnect()

if __name__ == "__main__":
    main()
