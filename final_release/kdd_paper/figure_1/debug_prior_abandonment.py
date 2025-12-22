import numpy as np
import matplotlib.pyplot as plt

def simulate_abandonment():
    # Constants
    DIM = 1 # Simplified 1D for clarity
    LAMBDA = 1.0
    
    # Setup
    alphas = [0.1, 0.5, 1.0, 2.0]
    
    print(f"{'Alpha':<6} | {'Steps to Switch':<15} | {'Explanation'}")
    print("-" * 60)
    
    for alpha in alphas:
        # Initialize Two Arms
        # Arm A: The "Favorite" (High Prior)
        # Arm B: The "Underdog" (Low Prior)
        
        # Priors (Mean)
        mu_A = 0.8
        mu_B = 0.5
        
        # Initial Uncertainty (Identical)
        # A_inv = 1/lambda = 1.0
        # x = 1.0
        # var = x * A_inv * x = 1.0
        # std = 1.0
        std_A = 1.0
        std_B = 1.0
        
        # State Tracking
        A_inv_A = 1.0
        b_A = mu_A # Initial b to give mean=0.8
        
        steps = 0
        switched = False
        
        # Simulation Loop
        for t in range(1, 21):
            # Calculate UCBs
            # Arm A (Favorite)
            theta_A = A_inv_A * b_A
            curr_mu_A = theta_A * 1.0
            curr_std_A = np.sqrt(A_inv_A)
            ucb_A = curr_mu_A + alpha * curr_std_A
            
            # Arm B (Underdog) - Never pulled yet
            ucb_B = mu_B + alpha * std_B
            
            if ucb_B > ucb_A:
                print(f"{alpha:<6} | {t:<15} | Switched! (UCB_B {ucb_B:.2f} > UCB_A {ucb_A:.2f})")
                switched = True
                break
            
            # Pull A and get Disappointing Reward (0.4)
            # Update A
            reward = 0.4
            
            # RLS Update (1D)
            # A <- A + x*x' = A + 1
            # A_inv <- A_inv - ... (Sherman Morrison)
            # For 1D: A_new = A_old + 1. A_inv_new = 1 / (1/A_inv_old + 1)
            A_old = 1.0 / A_inv_A
            A_new = A_old + 1.0
            A_inv_A = 1.0 / A_new
            
            # b <- b + r*x = b + 0.4
            b_A += reward
            
            steps += 1
            
        if not switched:
             print(f"{alpha:<6} | {'Never':<15} | Stuck on A (UCB_A {ucb_A:.2f} > UCB_B {ucb_B:.2f})")

if __name__ == "__main__":
    simulate_abandonment()
