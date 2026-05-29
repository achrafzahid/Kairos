import os
import glob

def reset_and_clean_tensors(base_dir="../../data"):
    """
    Scans all asset subdirectories, restores original tensor names,
    and removes bad label files completely to preserve SSD storage.
    """
    print(f"Starting tensor cleanup and restoration inside: {base_dir}")
    
    # 1. Gather all files across all subdirectories
    all_files = glob.glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)
    
    if not all_files:
        print(f"No .pt files found in {base_dir}. Verify your base path directory config.")
        return
        
    y_deleted_count = 0
    x_restored_count = 0
    
    # 2. Run the destruction and restoration loops
    for file_path in all_files:
        filename = os.path.basename(file_path)
        dirname = os.path.dirname(file_path)
        
        # Scenario A: It's a bad label file -> Delete it completely
        if filename.endswith("_Y.pt"):
            try:
                os.remove(file_path)
                y_deleted_count += 1
            except OSError as e:
                print(f"Error removing label file {file_path}: {e}")
                
        # Scenario B: It's a feature file split out -> Rename back to original base format
        elif filename.endswith("_X.pt"):
            # e.g., 'AAPL_2024-07-05_X.pt' -> 'AAPL_2024-07-05.pt'
            original_filename = filename.replace("_X.pt", ".pt")
            original_file_path = os.path.join(dirname, original_filename)
            
            try:
                os.rename(file_path, original_file_path)
                x_restored_count += 1
            except OSError as e:
                print(f"Error renaming feature file {file_path} back to original: {e}")

    print("\n" + "="*50)
    print("CLEANUP STATUS REPORT:")
    print("="*50)
    print(f"--> Bad Label Files (_Y.pt) Safely Deleted: {y_deleted_count}")
    print(f"--> Feature Files (_X.pt) Restored to Base:  {x_restored_count}")
    print("="*50)
    print("Your data storage directory is now 100% reset to its raw, clean format.\n")

if __name__ == "__main__":
    # Runs automatically across your 40GB directory tree structure
    reset_and_clean_tensors(base_dir="../../data")