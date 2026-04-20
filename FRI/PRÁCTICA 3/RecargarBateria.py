from robobosim.RoboboSim import RoboboSim 
from robobopy.Robobo import Robobo          
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color
from Behavior import Behavior

class RecargarBateria(Behavior):
    def __init__(self, robobo, supress_list, params, velocidad=15, kp=0.8, ki=0.01, kd=0.3):
        super().__init__(robobo, supress_list, params)
        self.velocidad = velocidad
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.xgoal = 50   
        self.zgoal = 50   
        
        self.prev_error = 0
        self.integral = 0

        self.robobo.setActiveBlobs(True, False, False, False)
        self.robobo.whenATapIsDetected(self.tapDetectedCallback)
        self.tap_detected = False

    def tapDetectedCallback(self):
        self.tap_detected = True

    def take_control(self):
        if self.tap_detected:
            self.tap_detected = False       # reseteamos la variable para la próxima vez
            return True
        return False

    def action(self):
        self.is_supressed = False
        for behavior in self.supress_list:
            behavior.is_supressed = True

        self.prev_error = 0
        self.integral = 0
        
        self.robobo.moveTiltTo(105, 5, True)

        self.robobo.sayText("RECARGANDO")
        while not self.is_supressed:
            self.robobo.wait(0.4)
            color = self.robobo.readColorBlob(Color.RED)
            ir = self.robobo.readIRSensor(IR.FrontC)

            if color and color.size > 0:
                if 45 <= color.posx <= 55:            # si está centrado (margen 45-55)
                    if ir < self.zgoal:
                        zerror = self.zgoal - ir
                        speed = round(zerror * self.kp)
                        self.robobo.moveWheels(speed, speed)
                    else:
                        self.robobo.stopMotors()
                        break 
                else:
                    # PID para centrado horizontal
                    xerror = self.xgoal - color.posx if abs(self.xgoal - color.posx) > 5 else 0
                    der = xerror - self.prev_error
                    self.integral += xerror
                    
                    xcorrection = round(self.kp * xerror + self.integral * self.ki + der * self.kd)
                    self.prev_error = xerror
                    
                    self.robobo.moveWheels(-xcorrection, xcorrection)
            else:
                # si no ve el color, lo busca girando
                self.robobo.moveTiltTo(95, 5)
                self.robobo.moveWheels(5, -5)

        # Liberar otros comportamientos al terminar
        for behavior in self.supress_list:
            behavior.is_supressed = False


            

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
    IP = "localhost"

    # sim = RoboboSim(IP) # conexión al simulador
    # sim.connect()

    robobo = Robobo(IP) # conexión al robobo
    robobo.connect()


    # recargar_bateria()

    while True:
        robobo.moveTiltTo(105, 5, True)
        robobo.moveWheelsByTime(10, -10, 3.25)
        robobo.moveWheelsByTime(15, 15, 7)
        robobo.whenATapIsDetected(recargar_bateria())
