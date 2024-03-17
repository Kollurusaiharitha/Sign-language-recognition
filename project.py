import cv2
import os

def convert_videos_to_frames(video_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all video files in the input folder
    for file_name in os.listdir(video_folder):
        if file_name.endswith('.MOV'):
            video_path = os.path.join(video_folder, file_name)

            # Open the video file
            video_capture = cv2.VideoCapture(video_path)

            # Check if the video file was successfully opened
            if not video_capture.isOpened():
                print(f"Failed to open {file_name}")
                continue

            # Read and save frames from the video
            frame_count = 0
            while True:
                # Read a single frame from the video
                ret, frame = video_capture.read()

                # Break if no frame is retrieved
                if not ret:
                    break

                # Generate the output file path
                output_file_name = f"{file_name}_{frame_count}.jpg"
                output_file_path = os.path.join(output_folder, output_file_name)

                # Save the frame as an image file
                cv2.imwrite(output_file_path, frame)

                frame_count += 1

            # Release the video capture object
            video_capture.release()

            print(f"Converted {frame_count} frames from {file_name}")

# Example usage
video_folder =r"C:\Haritha\MCA project\Adjectives\1. loud"# Folder containing the input videos
output_folder =r"C:\Haritha\MCA project\Frames\Loud" # Folder to save the output frames


print(f"Video Folder: {video_folder}")
print(f"Output Folder: {output_folder}")





convert_videos_to_frames(video_folder, output_folder)

