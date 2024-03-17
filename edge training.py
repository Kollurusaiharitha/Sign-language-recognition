import cv2
import numpy as np
from skimage.metrics import mean_squared_error
import os

def extract_signature(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_frame, threshold1=100, threshold2=200)
    
    return edges

def compare_signatures(sig1, sig2):
    return mean_squared_error(sig1.flatten(), sig2.flatten())

def select_best_frames(frames_folder, threshold=0.1, output_folder="output_frames"):
    os.makedirs(output_folder, exist_ok=True)

    frame_files = os.listdir(frames_folder)
    frame_files.sort()  # Ensuring frames are in order
    
    if len(frame_files) == 0:
        return

    prev_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    if prev_frame is None:
        print(f"Error loading frame: {os.path.join(frames_folder, frame_files[0])}")
        return

    prev_signature = extract_signature(prev_frame)
    frame_count = 1

    for frame_file in frame_files[1:]:
        curr_frame = cv2.imread(os.path.join(frames_folder, frame_file))
        if curr_frame is None:
            print(f"Error loading frame: {os.path.join(frames_folder, frame_file)}")
            continue

        curr_signature = extract_signature(curr_frame)
        mse = compare_signatures(prev_signature, curr_signature)
        
        if mse > threshold:
            output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, curr_frame)
            prev_signature = curr_signature
            frame_count += 1

# Example usage:
frames_folder =r"C:\Haritha\MCA project\Frames\Beautiful"
output_folder = r"C:\Haritha\MCA project\Edge based training\Beautiful"
select_best_frames(frames_folder, threshold=0.1, output_folder=output_folder)
