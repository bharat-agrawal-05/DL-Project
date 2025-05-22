import os

def delete_file(file_path):
    """
    Deletes a file if it exists.

    :param file_path: Path to the file to be deleted.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} deleted successfully.")
        else:
            print(f"File {file_path} does not exist.")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
    
for i in range(500, 20000):
    file_path = f"dataset/sa_{i}.jpg"
    delete_file(file_path)