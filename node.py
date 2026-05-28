import numpy as np

class LearningNode:
    def __init__(self, node_id, initial_T=2.0, initial_S=0.5,
                 eta1=3.0, eta2=1.0, eta3=0.1, #learning rates
                 x=0.0, y=0.0, z=0.0): #spatial position
        self.id = node_id #unique identifier
        self.T = initial_T #temperature
        self.S = initial_S #salinity
        self.eta1 = eta1 #thermal forcing strength
        self.eta2 = eta2 #salinity forcing strength
        self.eta3 = eta3 #diffusivity
        self.x = x #longitude-like position
        self.y = y #latitude-like position
        self.z = z #depth-like position
        self.step_count = 0 #step count
        self.min_psi = -2.0 #range
        self.max_psi = 4.0 #range
        self.last_dpsi_dt = 0.0 #last clean dΨ/dt from RK4, no noise, for display

    @property
    def psi(self): #psi= T-S overturning circulation strength
        return self.T - self.S

    @property
    def regime(self): #meridonal overturning strength
        return "TH" if self.psi > 0 else "SA"

    def step(self, inputs=None, dt=0.1, noise_sigma=0.5, model=None): #dt, noise, model tried 0.7 keeping 0.5 for now
        inp = sum(inputs) if inputs else 0.0 #sum of inputs
        eta2_eff = self.eta2 + inp #changes in salinity like melting fresh water

        def stommel(T, S): #stommel ODEs as a function for RK4 to call
            dT = self.eta1 - T * (1.0 + abs(T - S)) #stommels first box
            dS = eta2_eff - S * (self.eta3 + abs(T - S)) #stommels second box
            return dT, dS

        #RK4 slopes
        k1_T, k1_S = stommel(self.T, self.S)   #slope at start
        k2_T, k2_S = stommel(self.T + dt/2 * k1_T, self.S + dt/2 * k1_S) #slope at midpoint using k1
        k3_T, k3_S = stommel(self.T + dt/2 * k2_T, self.S + dt/2 * k2_S) #slope at midpoint using k2
        k4_T, k4_S = stommel(self.T + dt * k3_T,   self.S + dt * k3_S) #slope at end

        dT_dt = (k1_T + 2*k2_T + 2*k3_T + k4_T) / 6 #weighted average
        dS_dt = (k1_S + 2*k2_S + 2*k3_S + k4_S) / 6 #weighted average

        clean_next_T = self.T + dt * dT_dt #clean RK4 next T, no noise
        clean_next_S = self.S + dt * dS_dt #clean RK4 next S, no noise
        self.last_dpsi_dt = dT_dt - dS_dt #store clean dΨ/dt for display

        noise_T = np.random.normal(0, noise_sigma) #independent noise on each box
        noise_S = np.random.normal(0, noise_sigma) #independent noise on each box
        self.T = float(np.clip(clean_next_T + dt * noise_T, -10, 20)) #noisy state update
        self.S = float(np.clip(clean_next_S + dt * noise_S, -5, 15)) #noisy state update
        self.step_count += 1 #increase step count
        return clean_next_T, clean_next_S #return clean targets for dataset