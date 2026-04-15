#
#  Ejemplo de uso de threads para el desarrollo de arquitecturas reactivas (subsumida)
#

from robobopy.Robobo import Robobo

from behaviour_mod.go_to_wall import GoToWall
from behaviour_mod.avoid_wall import AvoidWall

import time

def main():
    robobo = Robobo("localhost")
    robobo.connect()

    # Diccionario que se pasará a los comportamientos
    # para que lo activen cuando se finalice la misión
    params = {"stop": False}

    # Creación de los comportamientos
    go_to_wall_behaviour = GoToWall(robobo, [], params)
    avoid_wall_behaviour = AvoidWall(robobo, [go_to_wall_behaviour], params)
    #find_color_behaviour = FindColor(robobo, [avoid_wall_behaviour, go_to_wall_behaviour], params, BlobColor.RED)

    # Se crea una Lista con los comportamientos (2 en este ejemplo)
    threads = [go_to_wall_behaviour, avoid_wall_behaviour]

    # Se inician todos los comportamientos
    go_to_wall_behaviour.start()
    avoid_wall_behaviour.start()

    # Se mantiene el hilo principal en espera
    # hasta que algún comportamiento marca
    # el objetivo como terminado
    while not params["stop"]:
        time.sleep(0.1)

    # Espera a que terminen todos los hilos
    for thread in threads:
        thread.join()

    robobo.disconnect()

if __name__ == "__main__":
    main()
