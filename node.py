import numpy as np

class LearningNode:
    def __init__(self, node_id, initial_T=2.0, initial_S=0.5, 
                 eta1=3.0, eta2=1.0, eta3=0.1): #learning rates
        self.id = node_id #unique identifier
        self.T = initial_T # Previously initial temperature
        self.S = initial_S # Previously initial salinity
        self.eta1 = eta1 #thermal forcing strength
        self.eta2 = eta2 #salinity forcing strength
        self.eta3 = eta3 #using the study salinity to temperature deffusivity
        self.step_count = 0 #step count
        self.min_psi = -2.0 #range 
        self.max_psi = 4.0 #range
        
@property
def psi(self): #psi= T-S overturning circulation strength
    return self.T - self.S

@property
def regime(self): #meridonal overturining strength
    return "TH" if self.psi > 0 else "SA"

def step(self, inputs=None, dt=0.1, noise_sigma=0.7, model=None): #dt, noise, model
    inp = sum(inputs) if inputs else 0.0 #sum of inputs
    eta2_eff = self.eta2 + inp #changes in salinity like melting fresh water

    if model is not None: #if theres a model:
        features = np.array([[self.T, self.S, self.eta1, eta2_eff, self.eta3]]) #temp, salinity, thermal and salinity forcing, and diffusivity
        dpsi_dt = float(model.predict(features)[0]) #Ψ = T - S, dT/dt - dS/dt
        dT_dt = self.eta1 - self.T * (1.0 + abs(self.T - self.S))  #stommels first box
        dS_dt = eta2_eff  - self.S * (self.eta3 + abs(self.T - self.S)) #stommels second box
    else:
        dT_dt = self.eta1 - self.T * (1.0 + abs(self.T - self.S)) #thermal changes dampend by weakening circulation 
        dS_dt = eta2_eff  - self.S * (self.eta3 + abs(self.T - self.S)) #similar for salinity but with diffusivity
        dpsi_dt = dT_dt - dS_dt #Ψ = T - S rate of change in circulation strength

    noise = np.random.normal(0, noise_sigma) #noise defined at def step
    dpsi_dt += noise #add noise

    self.T = float(np.clip(self.T + dt * dT_dt, -10, 20)) #euler integration prevents runaway
    self.S = float(np.clip(self.S + dt * dS_dt, -5, 15)) #euler integration prevents runaway

    self.step_count += 1 #increase step count
    return dpsi_dt