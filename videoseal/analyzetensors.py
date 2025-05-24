import torch
import torch.nn.functional as F
from itertools import combinations # For generating unique pairs

# --- Configuration ---
FILE_PATH = 'distinctMAIN.pt'

def load_tensors_from_file(file_path):
    """
    Loads tensors from a .pt file.
    Expects the file to contain a list of tensors or a single tensor
    where the first dimension represents individual tensors.
    """
    try:
        data = torch.load(file_path)
        if isinstance(data, list) and all(isinstance(t, torch.Tensor) for t in data):
            print(f"Successfully loaded {len(data)} tensors from '{file_path}' as a list.")
            return data
        elif isinstance(data, torch.Tensor):
            if data.ndim == 0: # Scalar tensor
                print(f"Loaded a single scalar tensor from '{file_path}'. Treating as a list with one tensor.")
                return [data]
            elif data.ndim > 0 and data.shape[0] > 1 : # Potentially a stack of tensors
                 # Check if it's likely a list of tensors saved as a single tensor
                 # This is a heuristic. If the first dimension is 50, it's a good candidate.
                if data.shape[0] == 50: # Or some other expected number if known
                    tensors_list = [data[i] for i in range(data.shape[0])]
                    print(f"Loaded a single tensor and split it into {len(tensors_list)} tensors along the first dimension.")
                    return tensors_list
                else: # It's a single tensor, not a stack of 50
                    print(f"Loaded a single tensor (shape: {data.shape}) from '{file_path}'. Treating as a list with one tensor.")
                    return [data] # Treat as a list containing one tensor
            else: # Single tensor with first dimension 1, or other cases
                print(f"Loaded a single tensor (shape: {data.shape}) from '{file_path}'. Treating as a list with one tensor.")
                return [data]
        else:
            print(f"Error: Expected '{file_path}' to contain a list of tensors or a single multi-item tensor.")
            print(f"Instead, found data of type: {type(data)}")
            return None
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading or processing '{file_path}': {e}")
        return None

def print_loaded_tensors(tensors_list):
    """
    Prints the loaded tensors with their indices.
    """
    if not tensors_list:
        print("No tensors to print.")
        return

    print("\n--- Displaying Loaded Tensors ---")
    for i, tensor in enumerate(tensors_list):
        print(f"\nTensor {i}:")
        print(f"Shape: {tensor.shape}")
        print(f"Data type: {tensor.dtype}")
        print(tensor)
        # Add a small separator for readability if tensors are large
        if i < len(tensors_list) - 1:
            print("-" * 20)
    print("--- End of Tensor Display ---")


def calculate_cosine_similarities(tensors_list):
    """
    Calculates and prints cosine similarity between every unique pair of tensors.
    """
    if not tensors_list or len(tensors_list) < 2:
        print("Not enough tensors to compare (need at least 2).")
        return {}

    num_tensors = len(tensors_list)
    print(f"\nCalculating cosine similarities for {num_tensors} tensors:")

    similarity_results = {}
    pair_indices = list(combinations(range(num_tensors), 2))

    for i, j in pair_indices:
        tensor_a = tensors_list[i]
        tensor_b = tensors_list[j]

        if tensor_a.ndim > 1:
            tensor_a_flat = tensor_a.flatten()
        else:
            tensor_a_flat = tensor_a
        
        if tensor_b.ndim > 1:
            tensor_b_flat = tensor_b.flatten()
        else:
            tensor_b_flat = tensor_b

        if tensor_a_flat.ndim == 0 or tensor_b_flat.ndim == 0:
            print(f"Skipping similarity between Tensor {i} and Tensor {j}: One or both are scalars after flattening.")
            similarity_results[(i, j)] = float('nan')
            continue
        
        if tensor_a_flat.shape != tensor_b_flat.shape:
            # If shapes mismatch after flattening, cosine similarity is not directly applicable
            # unless you have a specific strategy (e.g., padding, truncation, or it's an error).
            print(f"Skipping similarity between Tensor {i} (flat shape {tensor_a_flat.shape}) and Tensor {j} (flat shape {tensor_b_flat.shape}): Mismatched shapes after flattening.")
            similarity_results[(i,j)] = float('nan')
            continue


        norm_a = torch.linalg.norm(tensor_a_flat.float()) # Ensure float for norm
        norm_b = torch.linalg.norm(tensor_b_flat.float())

        if norm_a == 0 or norm_b == 0:
            similarity = 0.0 if not (norm_a == 0 and norm_b == 0) else float('nan')
        else:
            # Ensure tensors are float for cosine_similarity
            similarity_tensor = F.cosine_similarity(tensor_a_flat.float().unsqueeze(0), tensor_b_flat.float().unsqueeze(0), dim=1)
            similarity = similarity_tensor.item()

        similarity_results[(i, j)] = similarity
        print(f"Similarity between Tensor {i} (shape {tensor_a.shape}) and Tensor {j} (shape {tensor_b.shape}): {similarity:.4f}")

    print(f"\nCalculated {len(similarity_results)} unique pair similarities.")
    return similarity_results

if __name__ == "__main__":
    # 1. Load the tensors
    tensors = load_tensors_from_file(FILE_PATH)

    if tensors:
        # User mentioned 50 tensors, let's check
        if len(tensors) != 50:
            print(f"\nWarning: Expected 50 tensors based on description, but loaded {len(tensors)}.")
        
        # 2. Print the loaded tensors
        #print_loaded_tensors(tensors) # New function call

        # 3. Calculate and print similarities
        results = calculate_cosine_similarities(tensors)
        
        # Optional: Further processing with 'results'
        # if results:
        #     sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
        #     print("\nTop 5 most similar pairs:")
        #     for k, v in sorted_results[:5]:
        #         if not torch.isnan(torch.tensor(v)):
        #             print(f"Pair {k}: {v:.4f}")