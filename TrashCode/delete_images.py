import os

def delete(file_path):
    """
    Deletes a file if it exists.
    
    Args:
        file_path (str): The path to the file to be deleted.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")
        else:
            print(f"{file_path} does not exist.")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

for i in range(60000, 70000):
    file_path = f"./test/sa_{i}.jpg"
    delete(file_path)