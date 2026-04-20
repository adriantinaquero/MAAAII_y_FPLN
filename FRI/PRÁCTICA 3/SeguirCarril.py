from robobosim.RoboboSim import RoboboSim   
from robobopy.Robobo import Robobo          
from robobopy.utils.IR import IR
from Behavior import Behavior


class SeguirCarril(Behavior):
    def __init__(self, robobo, supress_list, params, velocidad=15, kp=0.8, ki=0.01, kd=0.3):
            super().__init__(robobo, supress_list, params)
            self.velocidad = velocidad
            self.kp = kp
            self.ki = ki
            self.kd = kd
            
            self.prev_error = 0
            self.integral = 0
    def take_control(self):     # en principio siempre está activo
        if not self.is_supressed:
            return True 

    def action(self):
            self.supress = False
            for behavior in self.supress_list:
                behavior.is_supressed = True

            self.robobo.moveWheels(5, 5)
            self.robobo.sayText("SIGUIENDO")
            # ir_izq = self.robobo.readIRSensor(IR.FrontLL)
            # ir_der = self.robobo.readIRSensor(IR.FrontRR)

            # error = ir_der - ir_izq
            # P = self.kp * error

            # self.integral += error
            # I = self.ki * self.integral

            # D = self.kd * (error - self.prev_error)
            # self.prev_error = error

            # correccion = round(P + I + D)

            # vel_izq = self.velocidad + correccion
            # vel_der = self.velocidad - correccion

            # vel_izq = max(min(vel_izq, 40), -40)
            # vel_der = max(min(vel_der, 40), -40)

            # self.robobo.moveWheels(vel_izq, vel_der)

            for behavior in self.supress_list:
                behavior.is_supressed = False





def seguir_carril(velocidad: int = 15, kp: float = 0.8, ki: float = 0.01, kd: float = 0.3, prev_error = 0, integral = 0):

    while True:
        # ir_izq = robobo.readIRSensor(IR.FrontLL)
        ir_der = robobo.readIRSensor(IR.FrontRR)

        error = ir_der - 60      # 60 es la distancia a la que debe mantenerse de la línea del carril
        P = kp * error
        
        integral += error
        I = ki * integral
        
        D = kd * (error - prev_error)
        prev_error = error

        correccion = round(P + I + D)

        vel_izq = velocidad - correccion
        vel_der = velocidad + correccion

        # # limitamos velocidad
        # vel_izq = max(min(vel_izq, 10), -10)
        # vel_der = max(min(vel_der, 10), -10)

        robobo.moveWheels(vel_izq, vel_der)





if __name__=="__main__":
    IP = "172.20.10.14"

    # sim = RoboboSim(IP) # conexión al simulador
    # sim.connect()

    robobo = Robobo(IP) # conexión al robobo
    robobo.connect()

    seguir_carril(velocidad=15, kp=0.024, ki=0.0001, kd=0.5)