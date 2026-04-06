import numpy as np

class LearningNode:
    def __init__(self, node_id, initial_state):
        self.id = node_id
        self.state = initial_state
        self.value = 0.0
        self.weight = 1.0
        self.threshold = 0.5
        self.min_state = -10
        self.max_state = 10

    def step(self, inputs=None):
        # Advance state (clamped)
        self.state = min(self.state + 1, self.max_state)

        # Compute delta from inputs
        delta = sum(inputs) if inputs else 0.0

        # Stable value update: tanh-bounded weighted blend
        raw = (
            self.value * 0.85 +
            np.tanh(delta * 0.1) * 5.0 +
            np.sin(self.state * 0.5) * self.weight
        )
        self.value = float(np.clip(raw, -1000, 1000))

        # Slowly adapt weight toward activity level
        activity = abs(delta)
        self.weight = float(np.clip(self.weight * 0.99 + activity * 0.01, 0.01, 10.0))

        # Threshold drifts toward median weight region
        self.threshold = float(np.clip(self.threshold * 0.98 + self.weight * 0.02, 0.01, 5.0))