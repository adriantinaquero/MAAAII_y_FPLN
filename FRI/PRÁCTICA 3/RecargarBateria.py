from robobopy.utils.IR import IR
from robobopy.utils.Color import Color
from Behavior import Behavior



class RecargarBateria(Behavior):
    def __init__(self, robobo, supress_list, params, velocidad=15, kp=0.5, ki=0.02, kd=0.1, kp2=0.1):
        super().__init__(robobo, supress_list, params)

        # variables para el PID
        self.velocidad = velocidad
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kp2 = kp2
        
        self.xgoal = 50   
        self.zgoal = 50   
        
        self.prev_error = 0
        self.integral = 0

        self.robobo.setActiveBlobs(True, False, False, False)
        self.robobo.whenATapIsDetected(self.tapDetectedCallback)
        self.tap_detected = False

    # si detecta un tap, tap_detected es True
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
        
        self.robobo.stopMotors()
        self.robobo.moveTiltTo(100, 5, True)

        while not self.is_supressed:
            self.robobo.wait(0.4)
            color = self.robobo.readColorBlob(Color.RED)    # va a buscar un blob rojo
            ir = self.robobo.readIRSensor(IR.FrontC)

            if color and color.size > 0:
                if 45 <= color.posx <= 55:            # si está centrado (margen 45-55)
                    if ir < self.zgoal:
                        zerror = self.zgoal - ir      # se va acercando hasta el
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
                    
                    xcorrection = round(self.kp2 * xerror + self.integral * self.ki + der * self.kd)
                    self.prev_error = xerror
                    
                    self.robobo.moveWheels(-xcorrection, xcorrection)
            else:
                # si no ve el color, lo busca girando
                self.robobo.moveTiltTo(100, 5)
                self.robobo.moveWheels(5, -5)

        for behavior in self.supress_list:
            behavior.is_supressed = False