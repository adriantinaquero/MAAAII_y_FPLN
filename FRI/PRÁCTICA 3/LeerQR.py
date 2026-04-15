from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from Behavior import Behavior        


# NO DEBERÍAMOS UTILIZAR MOVEWHEELSBYTIME, YA QUE BLOQUEA LOS HILOS DURANTE ESTE TIME


class LeerQR(Behavior):
    def __init__(self, robobo, supress_list, params):
        super().__init__(robobo, supress_list, params)
        self.ultimo_qr_visto = None

    def take_control(self):
        if not self.supress:
            qr = self.robobo.readQR()

            if qr and qr.distance > 25:
                self.ultimo_qr_visto = qr
                return True
            
            self.ultimo_qr_visto = None
            return False

    def action(self):
        self.supress = False
        for behavior in self.supress_list:
             behavior.is_supressed = True

        qr = self.ultimo_qr_visto
        
        if qr.id == "peligro izquierda":
            self.robobo.stopMotors()
            self.robobo.sayText("PELIGRO")
            
        elif qr.id == "autobus":
            self.robobo.setLedColorTo(LED.All, Color.GREEN)
            self.robobo.sayText("AUTOBUS")
            self.robobo.moveWheelsByTime(30, 30, 2)

        for behavior in self.supress_list:
            behavior.is_supressed = False
