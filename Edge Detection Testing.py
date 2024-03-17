import os
import random
import shutil

def select_frames(folder_path, signature_frames, percentage=20, output_path=None):
    # Get the list of all frames in the folder
    all_frames = os.listdir(folder_path)
    
    # Calculate the number of frames to select based on the given percentage
    num_frames_to_select = int(len(all_frames) * percentage / 100)
    
    # Filter out the signature frames from the list of all frames
    filtered_frames = [frame for frame in all_frames if frame not in signature_frames]
    
    # Make sure we have enough non-signature frames to select
    if len(filtered_frames) < num_frames_to_select:
        raise ValueError("Not enough non-signature frames available for selection.")
    
    # Randomly select frames from the filtered list
    selected_frames = random.sample(filtered_frames, num_frames_to_select)
    
    # Create the output folder if not provided
    if output_path is None:
        output_path = os.path.join(folder_path, "selected_frames")
    os.makedirs(output_path, exist_ok=True)

    # Copy the selected frames to the output folder
    for frame in selected_frames:
        frame_path = os.path.join(folder_path, frame)
        shutil.copy(frame_path, output_path)

    return selected_frames

# Example usage:
folder_path = r"C:\Haritha\MCA project\Edge based training\Beautiful"
output_path = r"C:\Haritha\MCA project\Edge based testing\Beautiful"  # Replace this with the path where you want to store the selected frames
signature_frames = ["frame1.jpg", "frame5.jpg", "frame10.jpg"]  # Add your list of signature frames here
selected_frames = select_frames(folder_path, signature_frames, output_path=output_path)
print(selected_frames)
