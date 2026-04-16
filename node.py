import numpy as np

class LearningNode:
    def __init__(self, node_id, initial_state): #initial values for the node
        self.id = node_id
        self.state = initial_state
        self.value = 0.0
        self.weight = 1.0
        self.threshold = 0.5
        self.min_state = -10
        self.max_state = 10

    def step(self, inputs=None, dt=0.1, noise_sigma=0.7, model=None): #noise of 0.7
        inp = sum(inputs) if inputs else 0.0

        if model is not None: #once trainded predict the change in state
            features = np.array([[self.state, self.value, inp, self.weight, self.threshold]])
            F = float(model.predict(features)[0])
        else:
            F = np.tanh(inp * self.weight - self.threshold) * 2.0

        noise = np.random.normal(0, noise_sigma)
        ds_dt = F + noise
        self.value = self.value + dt * ds_dt
        self.value = float(np.clip(self.value, -100, 100))
        self.state = int(np.clip(self.state + 1, self.min_state, self.max_state))

        return ds_dt