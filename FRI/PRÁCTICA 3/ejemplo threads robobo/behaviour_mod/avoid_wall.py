#
# Comportamiento que si ve una pared cerca, se para
#

from behaviour_mod.behaviour import Behaviour
from robobopy.utils.IR import IR
from robobopy.utils.Sounds import Sounds

class AvoidWall(Behaviour):
    def __init__(self, robot, supress_list, params):
        super().__init__(robot, supress_list, params)
        self.front_distance = 100 #Valor de IR para que se active
        self.goal = 75 #Valor de IR para que pare

    #Método que define cuándo se activa el comportamiento
    def take_control(self):
        if not self.supress:
            if self.robot.readIRSensor(IR.FrontC) >= self.front_distance:
                return True

    #Método que define qué hace el comportamiento
    def action(self):
        print("----> control: Avoid Wall")
        self.supress = False
        for bh in self.supress_list:
            bh.supress = True

        self.robot.sayText("Veo una pared cerca")
        speed = 5
        if self.robot.readIRSensor(IR.FrontR) >= self.goal:
            self.robot.moveWheels(speed,-speed)
        else:
            self.robot.moveWheels(-speed,speed)
        
        self.robot.sayText("Salgo hacia un lado")
        self.robot.wait(2)
        self.robot.moveWheels(speed,speed)
        self.robot.wait(5)
        self.robot.playSound(Sounds.LAUGH)
        self.robot.stopMotors()

        for bh in self.supress_list:
            bh.supress = False

         
