from state import State
from my_token import Token
import copy

class Transition(object):
    def __init__(self, action: str, dependency: str = None):
        self._action = action
        self._dependency = dependency

    @property
    def action(self):
        return self._action

    @property
    def dependency(self):
        return self._dependency

    def __str__(self):
        return f"{self._action}-{self._dependency}" if self._dependency else str(self._action)


class Sample(object):
    def __init__(self, state: State, transition: Transition):
        self._state = state
        self._transition = transition

    @property
    def state(self):
        return self._state

    @property
    def transition(self):
        return self._transition

    def state_to_feats(self, nbuffer_feats: int = 2, nstack_feats: int = 2):
        feats = []
        stack = self._state.S
        buffer = self._state.B

        # Palabras de la Stack (de arriba a abajo)
        for i in range(nstack_feats):
            if len(stack) > i:
                feats.append(stack[-(i+1)].form)
            else:
                feats.append("<PAD>")

        # Palabras del Buffer
        for i in range(nbuffer_feats):
            if len(buffer) > i:
                feats.append(buffer[i].form)
            else:
                feats.append("<PAD>")

        # POS de la Stack
        for i in range(nstack_feats):
            if len(stack) > i:
                feats.append(stack[-(i+1)].upos)
            else:
                feats.append("<PAD>")

        # POS del Buffer
        for i in range(nbuffer_feats):
            if len(buffer) > i:
                feats.append(buffer[i].upos)
            else:
                feats.append("<PAD>")

        return feats


class ArcEager():
    LA = "LEFT-ARC"
    RA = "RIGHT-ARC"
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def create_initial_state(self, sent: list['Token']) -> State:
        # En Arc-Eager el ROOT empieza en la pila y el resto en el buffer
        return State([sent[0]], sent[1:], set([]))
    
    def final_state(self, state: State) -> bool:
        return len(state.B) == 0

    def LA_is_valid(self, state: State) -> bool:
        if not state.S or not state.B:
            return False
        last_s = state.S[-1].id
        # Precondición: El elemento de la pila no puede ser ROOT (id 0) 
        # y no puede tener ya un padre
        if last_s == 0:
            return False
        for _, _, dependent in state.A:
            if dependent == last_s:
                return False
        return True

    def LA_is_correct(self, state: State, sent) -> bool:
        if not self.LA_is_valid(state):
            return False
        # Es correcto si el primer elemento del buffer es padre del tope de la pila
        s0 = state.S[-1].id
        b0 = state.B[0].id
        for head, _, dependent in self.gold_arcs(sent):
            if head == b0 and dependent == s0:
                return True
        return False
    
    def RA_is_valid(self, state: State) -> bool:
        if not state.S or not state.B:
            return False
        # Precondición: El elemento del buffer no puede tener ya un padre
        b0 = state.B[0].id
        for _, _, dependent in state.A:
            if dependent == b0:
                return False
        return True

    def RA_is_correct(self, state: State, sent) -> bool:
        if not self.RA_is_valid(state):
            return False
        s0 = state.S[-1].id
        b0 = state.B[0].id
        for head, _, dependent in self.gold_arcs(sent):
            if head == s0 and dependent == b0:
                return True
        return False

    def REDUCE_is_valid(self, state: State) -> bool:
        if not state.S:
            return False
        # Precondición: El elemento de la pila debe tener ya un padre
        s0 = state.S[-1].id
        for _, _, dependent in state.A:
            if dependent == s0:
                return True
        return False

    def REDUCE_is_correct(self, state: State, sent) -> bool:
        if not self.REDUCE_is_valid(state):
            return False
        s0 = state.S[-1].id
        # Es correcto reducir si s0 ya tiene su padre (valid) 
        # Y no tiene hijos pendientes en el buffer
        for head, _, dependent in self.gold_arcs(sent):
            if head == s0:
                if any(t.id == dependent for t in state.B):
                    return False
        return True

    def oracle(self, sent: list['Token']) -> list['Sample']:
        state = self.create_initial_state(sent) 
        samples = []
        gold = self.gold_arcs(sent)

        while not self.final_state(state):
            transition = None
            
            if self.LA_is_correct(state, sent):
                dep = next(arc[1] for arc in gold if arc[0] == state.B[0].id and arc[2] == state.S[-1].id)
                transition = Transition(self.LA, dep)
            
            elif self.RA_is_correct(state, sent):
                dep = next(arc[1] for arc in gold if arc[0] == state.S[-1].id and arc[2] == state.B[0].id)
                transition = Transition(self.RA, dep)
                
            elif self.REDUCE_is_correct(state, sent):
                transition = Transition(self.REDUCE)
                
            else:
                transition = Transition(self.SHIFT)

            # CRÍTICO: Guardar una copia profunda para que los samples no cambien
            samples.append(Sample(copy.deepcopy(state), transition))
            self.apply_transition(state, transition)

        return samples

    def apply_transition(self, state: State, transition: Transition):
        t = transition.action
        dep = transition.dependency
        
        if t == self.LA:
            b = state.B[0]
            s = state.S.pop()
            state.A.add((b.id, dep, s.id))

        elif t == self.RA:
            s = state.S[-1]
            b = state.B.pop(0)
            state.A.add((s.id, dep, b.id))
            state.S.append(b)

        elif t == self.REDUCE:
            state.S.pop()

        elif t == self.SHIFT:
            state.S.append(state.B.pop(0))

    def gold_arcs(self, sent: list['Token']) -> set:
        return {(t.head, t.dep, t.id) for t in sent if t.id != 0}