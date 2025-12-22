import numpy as np

def test_scaling():
    dim = 384
    N = 26223
    score = 0.05
    strength = 20.0
    
    # Simulate a diverse benchmark: average vector is small
    avg_vec = np.random.randn(dim)
    avg_vec /= np.linalg.norm(avg_vec)
    avg_vec *= 0.006 # Typical norm for N=26k
    
    # Simulate average covariance
    avg_cov = (1.0/dim) * np.eye(dim)
    
    # Current Logic
    A = strength * (np.eye(dim) + avg_cov)
    b = strength * score * avg_vec
    theta = np.linalg.inv(A) @ b
    mean = avg_vec.dot(theta)
    print(f"Current Logic: Mean Reward = {mean:.6f} (Target: {score})")
    
    # Proposed Logic: Scale b to hit the target
    norm_sq = np.linalg.norm(avg_vec)**2
    b_new = strength * score * (avg_vec / norm_sq)
    theta_new = np.linalg.inv(A) @ b_new
    mean_new = avg_vec.dot(theta_new)
    print(f"Proposed Logic: Mean Reward = {mean_new:.6f} (Target: {score})")

if __name__ == "__main__":
    test_scaling()
