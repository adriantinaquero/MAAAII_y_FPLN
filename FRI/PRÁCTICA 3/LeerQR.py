from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from Behavior import Behavior        


class LeerQR(Behavior):
    def __init__(self, robobo, supress_list, params):
        super().__init__(robobo, supress_list, params)
        self.ultimo_qr_visto = None

    def take_control(self):
        if not self.supress:
            qr = self.robobo.readQR()
            id = qr.id if qr.id != None else "NONE"
            last_id = self.ultimo_qr_visto.id if self.ultimo_qr_visto != None else "NONE"
            if qr and qr.distance > 25 and last_id != id:
                self.ultimo_qr_visto = qr
                return True
            
            return False

    def action(self):
        self.supress = False
        for behavior in self.supress_list:
            behavior.is_supressed = True
        self.robobo.stopMotors()
        qr = self.ultimo_qr_visto
        
        self.robobo.sayText(qr.id)

        for behavior in self.supress_list:
            behavior.is_supressed = False
