import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# --- Parameters ---
num_vectors = 50
dimensionality = 128

# --- Generate Vectors for Very Distinct Cosine Similarities within [0,1] ---

# Strategy: Create "sparse" vectors or vectors that point towards different "corners"
# of the [0,1] hypercube.
# This will naturally lead to more varied dot products.

vectors_distinct = torch.zeros(num_vectors, dimensionality, dtype=torch.float32)

# Approach 1: One-hot-like vectors with some overlap and small noise
# This aims for similarities that can be close to 0.
num_active_dims_per_vector = 10 # Number of '1's or high values in each vector (tune this)
overlap_factor = 0.2 # Controls how much vectors overlap in their active dimensions
                     # 0.0 means no overlap, leading to 0 similarity if active sets are disjoint.
                     # 1.0 means full overlap, leading to high similarity.

all_possible_indices = list(range(dimensionality))

for i in range(num_vectors):
    # Determine the active dimensions for this vector
    if i == 0:
        # First vector: just pick random active dimensions
        active_indices = np.random.choice(all_possible_indices, num_active_dims_per_vector, replace=False)
    else:
        # For subsequent vectors, create some overlap with previous vectors
        prev_active_indices = np.where(vectors_distinct[i-1] > 0.5)[0] # Get active dims from previous vector
        num_overlap = int(num_active_dims_per_vector * overlap_factor)
        num_new = num_active_dims_per_vector - num_overlap

        # Select some indices from previous active ones (if enough)
        overlap_indices = np.random.choice(prev_active_indices, min(len(prev_active_indices), num_overlap), replace=False)

        # Select new, non-overlapping indices
        available_new_indices = list(set(all_possible_indices) - set(overlap_indices))
        new_indices = np.random.choice(available_new_indices, num_new, replace=False)

        active_indices = np.concatenate([overlap_indices, new_indices])
        np.random.shuffle(active_indices) # Shuffle to mix things up

    # Set active dimensions to 1.0 or a value between 0.5 and 1.0
    vectors_distinct[i, active_indices] = 1.0 # torch.rand(len(active_indices)) * 0.5 + 0.5

    # Add a small amount of noise to all elements to ensure min/max range and prevent exact zeros
    # This will make min_val > 0 and max_val < 1 slightly after clamping.
    vectors_distinct[i, :] += torch.rand(dimensionality) * 0.01


# Ensure all elements are strictly within [0,1]
vectors_distinct = torch.clamp(vectors_distinct, 0.0, 1.0)


print("--- Generated Distinct Vectors (elements already in [0, 1]) ---")
print("Min value (actual):", vectors_distinct.min().item())
print("Max value (actual):", vectors_distinct.max().item())
print("Shape:", vectors_distinct.shape)
# print(vectors_distinct) # Uncomment to see the tensors

# --- Calculate Cosine Similarities ---
similarities = []
for i, j in combinations(range(num_vectors), 2):
    tensor_a = vectors_distinct[i]
    tensor_b = vectors_distinct[j]

    # Handle zero vectors if clamping resulted in any (though highly unlikely with added noise)
    norm_a = torch.linalg.norm(tensor_a)
    norm_b = torch.linalg.norm(tensor_b)

    if norm_a == 0 or norm_b == 0:
        sim = 0.0 if not (norm_a == 0 and norm_b == 0) else float('nan')
    else:
        sim = F.cosine_similarity(tensor_a.unsqueeze(0), tensor_b.unsqueeze(0), dim=1).item()

    similarities.append(sim)

print(f"\nCalculated {len(similarities)} unique pairwise cosine similarities.")

# --- Analyze the Distribution of Similarities ---
if similarities:
    similarities_np = np.array(similarities)
    print(f"Min similarity: {similarities_np.min():.4f}")
    print(f"Max similarity: {similarities_np.max():.4f}")
    print(f"Mean similarity: {similarities_np.mean():.4f}")
    print(f"Std Dev of similarities: {similarities_np.std():.4f}")

    plt.figure(figsize=(10, 6))
    # Adjust bins based on the number of unique similarities
    num_bins = min(50, int(np.sqrt(len(similarities_np)))) if len(similarities_np) > 0 else 10
    plt.hist(similarities_np, bins=num_bins, edgecolor='black')
    plt.title('Distribution of Pairwise Cosine Similarities (for custom [0,1] vectors)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
else:
    print("No similarities to analyze.")


# --- Save the final vectors ---
torch.save(vectors_distinct, 'distinctMAIN_very_distinct_sim.pt')

# If you need a list of 50 tensors (each row as a tensor):
list_of_final_tensors = [vectors_distinct[i] for i in range(vectors_distinct.shape[0])]
torch.save(list_of_final_tensors, 'distinctMAIN.pt')