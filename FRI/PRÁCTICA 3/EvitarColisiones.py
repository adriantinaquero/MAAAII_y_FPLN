from robobopy.utils.IR import IR
from Behavior import Behavior        
    

# NO DEBERÍAMOS UTILIZAR MOVEWHEELSBYTIME, YA QUE BLOQUEA LOS HILOS DURANTE ESTE TIME


class EvitarColisiones(Behavior):
    def __init__(self, robot, supress_list, params):
        super().__init__(robot, supress_list, params)

    def take_control(self):
        if not self.supress:
            if self.robobo.readIRSensor(IR.FrontLL) > 30:       # si algún sensor detecta algo muy cerca 
                return True
            return False

    def action(self):
        self.supress = False
        for behavior in self.supress_list:
             behavior.is_supressed = True

        self.robobo.moveWheelsByTime(-10, -10, 2)   # se movería hacia atras y giraría

        for behavior in self.supress_list:
            behavior.is_supressed = False