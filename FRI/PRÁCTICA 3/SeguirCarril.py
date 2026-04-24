from robobopy.Robobo import Robobo          
from Behavior import Behavior
from robobopy.utils.IR import IR
from robobopy_videostream.RoboboVideo import RoboboVideo 
from robobopy.Robobo import Robobo 
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SeguirCarril(Behavior):
    def __init__(self, robobo, supress_list, params, videoStream, velocidad=30, kp=0.1, ki=0, kd=0):
            super().__init__(robobo, supress_list, params)
            self.velocidad = velocidad
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.videoStream = videoStream
            self.prev_error = 0
            self.integral = 0
            robobo.moveTiltTo(105, 20)


    def take_control(self):     # en principio siempre está activo, es el comportamiento más básico
        if not self.is_supressed:
            return True


    def action(self):

        # self.videoStream.connect()
        while self.supress == False:
            self.robobo.wait(0.1)
            full_frame = self.videoStream.getImage()
            full_frame = cv2.flip(full_frame, 1)
            frame = full_frame[440:640]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            h, w = np.shape(gray)

            centers = []

            image_center = w / 2

            for i in range(0, h, 2):
                xs = np.where((gray[i] > 240))[0]
                xs_izq = xs[xs < image_center]
                xs_der = xs[xs >= image_center]
                if len(xs_der) > 0:
                    centers.append(np.min(xs_der))
                else:
                    centers.append(image_center + 150)
                         

            lane_center = np.mean(centers)

            cv2.line(full_frame, (w // 2, 0), (w // 2, 640), (0, 0, 255))
            for (x, h) in zip(centers, range(0, h, 2)):
                cv2.circle(full_frame, (int(x), int(h + 440)), 3, (0, 255, 0), -1)
    
            cv2.imshow("robobo", full_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # (gray > 240).astype(np.uint8) * 255
            error = int(lane_center - 410)
            xerror = error
            der = xerror - self.prev_error
            self.prev_error = xerror 
            self.integral += xerror
            xcorrection = round(self.kp * xerror + self.integral * self.ki + der * self.kd)
            # print(error, lane_center, xcorrection)
            self.robobo.moveWheels(self.velocidad - xcorrection, self.velocidad + xcorrection)
        self.robobo.stopMotors()
        cv2.destroyAllWindows()
        # self.videoStream.disconnect()





# función para testear si funciona el comportamiento
def seguir_carril(velocidad: int = 5, kp: float = 0.025, ki: float = 0.01, kd: float = 0.3, prev_error = 0, integral = 0):
    videoStream.connect()
    prev_err = 0 # Inicializamos las variables necesarias para el PID
    integral = 0
    try:
        while True:
            robobo.wait(0.1)
            full_frame = videoStream.getImage()
            full_frame = cv2.flip(full_frame, 1)
            frame = full_frame[440:640]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            h, w = np.shape(gray)

            centers = []

            image_center = w / 2

            for i in range(0, h, 2):
                xs = np.where((gray[i] > 240))[0]
                xs_izq = xs[xs < image_center]
                xs_der = xs[xs >= image_center]
                if len(xs_der) > 0:
                    centers.append(np.min(xs_der))
                else:
                    centers.append(image_center + 150)
                         

            lane_center = np.mean(centers)

            cv2.line(full_frame, (w // 2, 0), (w // 2, 640), (0, 0, 255))
            for (x, h) in zip(centers, range(0, h, 2)):
                cv2.circle(full_frame, (int(x), int(h + 440)), 3, (0, 255, 0), -1)
    
            cv2.imshow("robobo", full_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # (gray > 240).astype(np.uint8) * 255
            error = int(lane_center - 370)
            xerror = error
            der = xerror - prev_err
            prev_err = xerror 
            integral += xerror
            xcorrection = round(kp * xerror + integral * ki + der * kd)
            print(error, lane_center, xcorrection)
            robobo.moveWheels(velocidad - xcorrection, velocidad + xcorrection)

    finally:
        robobo.stopMotors()
        robobo.disconnect()
        cv2.destroyAllWindows() 
        videoStream.disconnect()

# def seguir_carril(velocidad: int = 5, kp: float = 0.025, ki: float = 0.01, kd: float = 0.3, prev_error = 0, integral = 0):
#     videoStream.connect()
#     prev_err = 0 # Inicializamos las variables necesarias para el PID
#     integral = 0
#     try:
#         while True:
#             robobo.wait(0.1)
#             frame = cv2.flip(videoStream.getImage()[340:640, :, :], 1)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             h, w = np.shape(gray)

#             centers = []

#             image_center = w / 2

#             for i in range(0, h, 2):
#                 xs = np.where((gray[i] > 127))[0]
                
#                 if len(xs) > 0:
#                     xs_izq = xs[xs < image_center]
#                     xs_der = xs[xs >= image_center]
#                     if len(xs_der) > 0 and len(xs_izq) > 0:
#                         center = (np.max(xs_izq) + np.min(xs_der)) // 2
#                         centers.append(np.min(xs_der))

#             lane_center = np.mean(centers)

#             cv2.line(frame, (w // 2, 0), (w // 2, 640), (0, 0, 255))
#             for (x, h) in zip(centers, range(0, h, 2)):
#                 cv2.circle(frame, (int(x), int(h)), 3, (0, 255, 0), -1)
    
#             cv2.imshow("robobo", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
            
#             error = int(lane_center - 350)
#             # En otro caso activamos el PID
#             xerror = error # Limitamos ligeramente el rango objetivo para evitar bloqueos irresolubles
#             der = xerror - prev_err
#             prev_err = xerror # Calculamos los valores necesarios para el PID
#             integral += xerror
#             xcorrection = round(kp * xerror + integral * ki + der * kd)
#             print(error, lane_center, xcorrection)
#             robobo.moveWheels(velocidad - xcorrection, velocidad + xcorrection) # Aplicamos la correcion del PID {Si la corre

#     finally:
#         robobo.stopMotors()
#         robobo.disconnect()
#         cv2.destroyAllWindows() 
#         videoStream.disconnect()



if __name__=="__main__":
    IP = "localhost"
    # IP = "172.20.10.2"
    # sim = RoboboSim(IP) # conexión al simulador
    # sim.connect()
    
    videoStream = RoboboVideo(IP)
    robobo = Robobo(IP)
    robobo.connect()
    robobo.startStream()
    robobo.moveTiltTo(105, 20)
    seguir_carril(velocidad=30, kp=0.075, ki=0, kd=0)
