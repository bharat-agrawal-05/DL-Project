import torch

def generate_regular_simplex(n: int) -> torch.Tensor:
    """
    Returns (n+1) points in R^n forming a regular simplex:
      - each vector has unit norm
      - pairwise dot-products = -1/n
    """
    eye = torch.eye(n + 1)
    centroid = eye.mean(dim=0, keepdim=True)          # (1, n+1)
    V = eye - centroid                                 # (n+1, n+1)
    V = V[:, :-1]                                      # (n+1, n)
    V = V / V.norm(dim=1, keepdim=True)
    return V  # shape: (n+1, n)

# Generate 50 vectors in R^49 and embed into R^128
simplex49 = generate_regular_simplex(49)               # (50, 49)
zeros = torch.zeros((simplex49.size(0), 128 - 49))     # (50, 79)
embed128 = torch.cat([simplex49, zeros], dim=1)        # (50, 128)

# Optional random rotation
Q, _ = torch.qr(torch.randn(128, 128))                 # Q is orthogonal
vectors128 = embed128 @ Q                              # (50, 128)
vectors128 = vectors128 / vectors128.norm(dim=1, keepdim=True)  # normalize

# Display the actual tensor
#save vectors128 to a .pt file
torch.save(vectors128, 'vectors128.pt')
