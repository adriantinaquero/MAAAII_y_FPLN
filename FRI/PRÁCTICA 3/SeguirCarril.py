from robobosim.RoboboSim import RoboboSim   
from robobopy.Robobo import Robobo          
from Behavior import Behavior
from robobopy.utils.IR import IR
from robobopy_videostream.RoboboVideo import RoboboVideo 
from robobopy.Robobo import Robobo 
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    videoStream.connect()
    prev_err = 0 # Inicializamos las variables necesarias para el PID
    integral = 0
    try:
        while True:
            robobo.wait(0.4)
            frame = videoStream.getImage()[140:340, :, :]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = np.shape(gray)

            centers = []

            image_center = w / 2

            for i in range(0, h, 2):
                xs = np.where((gray[i] > 127))[0]
                
                if len(xs) > 0:
                    xs_izq = xs[xs < image_center]
                    xs_der = xs[xs >= image_center]
                    if len(xs_der) > 0 and len(xs_izq) > 0:
                        center = (np.max(xs_izq) + np.min(xs_der)) // 2
                        centers.append(center)

            lane_center = np.mean(centers)

            # cv2.line(frame, (w // 2, 0), (w // 2, 640), (0, 0, 255))
            # for (x, h) in zip(centers, range(0, h, 2)):
            #     cv2.circle(frame, (int(x), int(h)), 3, (0, 255, 0), -1)
    
            # cv2.imshow("robobo", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            error = int(lane_center - image_center)
            print(error)
            if error in range(-10, 10): # En caso de estar en el rango adecuado activamos el controlador proporcional
                robobo.moveWheels(5, 5)
            else: # En otro caso activamos el PID
                xerror = error # Limitamos ligeramente el rango objetivo para evitar bloqueos irresolubles
                der = xerror - prev_err
                prev_err = xerror # Calculamos los valores necesarios para el PID
                integral += xerror
                xcorrection = round(kp * xerror + integral * ki + der * kd)
                robobo.moveWheels(-xcorrection, xcorrection) # Aplicamos la correcion del PID {Si la corre
    finally:
        cv2.destroyAllWindows() 
        videoStream.disconnect()





if __name__=="__main__":
    IP = "172.20.10.14"

    # sim = RoboboSim(IP) # conexión al simulador
    # sim.connect()
    
    videoStream = RoboboVideo("localhost")
    robobo = Robobo(IP) # conexión al robobo
    robobo.connect()
    robobo.startStream()
    robobo.moveTiltTo(105, 20)
    seguir_carril(velocidad=5, kp=0.2, ki=0.02, kd=0)
