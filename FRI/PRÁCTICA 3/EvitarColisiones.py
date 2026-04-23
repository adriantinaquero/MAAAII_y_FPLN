from robobopy.utils.IR import IR
from Behavior import Behavior        
    


class EvitarColisiones(Behavior):
    def __init__(self, robot, supress_list, params):
        super().__init__(robot, supress_list, params)

    def take_control(self):
        if not self.supress:
            if self.robobo.readIRSensor(IR.FrontC) > 30:      # el sensor central delantero detecta algo muy cerca
                return True
            return False

    def action(self):
        self.supress = False
        for behavior in self.supress_list:
             behavior.is_supressed = True

        # el robobo retrocede, gira un poco y avanza, bordenado el obstáculo
        self.robobo.moveWheelsByTime(-10, -10, 2)
        self.robobo.moveWheelsByTime(15, -15, 1)
        self.robobo.moveWheelsByTime(10, 10, 4)
        self.robobo.moveWheelsByTime(-15, 15, 1)

        for behavior in self.supress_list:
            behavior.is_supressed = False