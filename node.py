class LearningNode:
    def __init__(self, node_id, initial_state):
        self.id = node_id
        self.state = initial_state  
        self.value = 0.0           
        self.weight = 1.0
        self.threshold = 0.5 #automatically but should change????
        self.min_state = -10
        self.max_state = 10

    #trans
    def step(self, inputs=None):
        self.state += 1 #state transition 
        if self.state > self.max_state:
            self.state = self.max_state 

      
        delta = 1.0  
        if inputs:
            delta += sum(inputs)*1  
        self.value += delta * self.weight #user input change