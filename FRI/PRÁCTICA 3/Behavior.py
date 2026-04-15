from threading import Thread
import time        



class Behavior(Thread):
    def __init__(self, robobo, supress_list, params, **kwargs):
        super().__init__(**kwargs)
        self.robobo = robobo
        self.supress_list = supress_list  # capas inferiores que este comportamiento puede anular
        self.params = params
        self.is_supressed = False

    def take_control(self): # devolvería true si el comportamiento debe actuar
        return False

    def action(self):  # define qué hace el comportamiento
        pass

    def supress(self):
        self.is_supressed = True
    
    # si algún comportamiento pone params["stop"] a True, se para termina la misión
    def run(self):
        while not self.params["stop"]:
            while not self.take_control() and not self.params["stop"]:
                time.sleep(0.01)
            if not self.params["stop"]:
                self.action()
            
    @property
    def supress(self):
        return self.is_supressed

    @supress.setter
    def supress(self, state):
        self.is_supressed = state

    def set_stop(self):
        self.params["stop"] = True

    def stopped(self):
        return self.params["stop"]
















