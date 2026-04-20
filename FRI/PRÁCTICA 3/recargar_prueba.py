from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color


def recargar_bateria(zgoal: int = 50, xgoal: int = 50, kp: float = 0.5):
    speed = 5
    prev_err = 0
    integral = 0
    kp2 = 0.15
    ki = 0.02
    kd = 0.1
    while True:
        robobo.wait(0.4)
        color = robobo.readColorBlob(Color.RED)
        ir = robobo.readIRSensor(IR.FrontC)
        if color.size > 0:
            if color.posx in range(45, 55):
                if ir < zgoal:
                    zerror = zgoal - ir
                    correction = round(zerror * kp)
                    speed = correction
                    robobo.moveWheels(speed, speed)
                else: robobo.stopMotors(); robobo.disconnect(); return ir, color.posx
            else:
                xerror = xgoal - color.posx if abs(xgoal - color.posx) > 5 else 0
                der = xerror - prev_err
                prev_err = xerror
                xcorrection = round(kp2 * xerror + integral * ki + der * kd)
                integral += xerror
                print(xcorrection, kp2 * xerror, integral * ki, der * kd, color.posx)
                robobo.moveWheels(xcorrection, -xcorrection)
        else:
            robobo.moveTiltTo(105, 5)
            robobo.moveWheels(5, -5)



if __name__=="__main__":

    robobo = Robobo("192.168.1.131")
    # robobo = Robobo("localhost")
    robobo.connect()

    recargar_bateria()