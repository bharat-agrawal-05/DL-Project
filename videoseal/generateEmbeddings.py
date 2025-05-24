import torch

# Create 10 tensors of size 128 with random float values between 0 and 1
tensors = [torch.rand(128) for _ in range(50)]

# Save the list of tensors to a file
torch.save(tensors, 'tensors.pt')
