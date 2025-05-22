# generate_binary_tensors.py
import torch
import numpy as np
import os

def generate_unique_binary_strings(num_strings: int, length: int) -> list[str]:
    """Generates a list of unique binary strings."""
    if num_strings > 2**length:
        raise ValueError(f"Cannot generate {num_strings} unique binary strings of length {length}. Max possible is {2**length}.")

    binary_strings = set()
    while len(binary_strings) < num_strings:
        # Generate a random binary string
        arr = np.random.randint(0, 2, size=length)
        binary_str = "".join(map(str, arr))
        binary_strings.add(binary_str)
    return list(binary_strings)

def main():
    num_messages = 100
    message_length = 32
    output_dir = "TokenPatternsDataset"
    output_filename = "tensorsBinary.pt"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"Generating {num_messages} unique binary messages of length {message_length}...")
    binary_messages_str = generate_unique_binary_strings(num_messages, message_length)

    # Convert list of strings to a list of lists of integers (0s and 1s)
    # then to a PyTorch tensor.
    # Storing them as strings in a list within the .pt file might be simpler if you prefer.
    # For direct use in your embedding function, string format is what it expects.
    # Let's save them as a list of strings.

    # Alternatively, if you need them as a tensor of numbers for some reason:
    # binary_messages_int_list = []
    # for msg_str in binary_messages_str:
    #     binary_messages_int_list.append([int(bit) for bit in msg_str])
    # messages_tensor = torch.tensor(binary_messages_int_list, dtype=torch.int8)

    # Saving as a list of strings, as your embedding function takes strings
    torch.save(binary_messages_str, output_path)

    print(f"Saved {len(binary_messages_str)} binary messages to {output_path}")

    # Optional: verify
    loaded_messages = torch.load(output_path)
    print(f"Verification: Loaded {len(loaded_messages)} messages. First message: {loaded_messages[0]}")

if __name__ == "__main__":
    main()