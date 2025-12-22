import numpy as np

def simulate_forgetting_vs_alpha():
    print(f"{'Parameter':<10} | {'Value':<6} | {'Steps to Switch':<15} | {'Explanation'}")
    print("-" * 70)
    
    # Scenario: Favorite (Prior 0.8) vs Underdog (Prior 0.5)
    # Favorite gets Reward 0.4 (Disappointing)
    
    def run_sim(alpha, gamma):
        # Initial State
        # Arm A (Favorite)
        mu_A = 0.8
        # Arm B (Underdog)
        mu_B = 0.5
        
        # Initial Matrices (Prior Strength 20.0)
        # A = I * lambda = 20.0
        # b = A * mu = 20.0 * 0.8 = 16.0
        A_A = 20.0
        b_A = 16.0
        
        # Arm B (Never pulled)
        # UCB_B is constant if we don't pull it
        # A_B = 20.0
        # b_B = 10.0
        # Std_B = sqrt(1/20) = 0.2236
        ucb_B = mu_B + alpha * 0.2236
        
        for t in range(1, 101):
            # Calculate UCB A
            A_inv_A = 1.0 / A_A
            theta_A = A_inv_A * b_A
            std_A = np.sqrt(A_inv_A)
            ucb_A = theta_A + alpha * std_A
            
            if ucb_B > ucb_A:
                return t
            
            # Update A (Favorite) with Reward 0.4
            # Forgetting: A <- gamma * A + x*x'
            # b <- gamma * b + r*x
            # x = 1.0
            A_A = gamma * A_A + 1.0
            b_A = gamma * b_A + 0.4
            
        return ">100"

    # 1. Vary Alpha (Gamma = 1.0)
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        steps = run_sim(alpha=alpha, gamma=1.0)
        print(f"{'Alpha':<10} | {alpha:<6} | {steps:<15} | Gamma=1.0 (No Forgetting)")
        
    print("-" * 70)
        
    # 2. Vary Gamma (Alpha = 0.1)
    for gamma in [1.0, 0.9, 0.8, 0.7]:
        steps = run_sim(alpha=0.1, gamma=gamma)
        print(f"{'Gamma':<10} | {gamma:<6} | {steps:<15} | Alpha=0.1 (Low Exploration)")

if __name__ == "__main__":
    simulate_forgetting_vs_alpha()
