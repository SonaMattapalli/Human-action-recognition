# Human-action-recognition
This is the assignment regarding Human Action Recognition using NVIDIA VLM Workflow done by team 10 from AI2(Sona Mattapalli[SE21UARI083], Manikanta Bollam[SE21UARI126], Laya Sunnapu[SE21UARI074])

# Action Detection with NVIDIA NEVA-22B

This project leverages the NVIDIA NEVA-22B API to identify specific actions (e.g., "walking," "standing") in videos. Users can upload two types of videos (synthetic and real) and compare the success rates of action detection in both.

## Features

•⁠  ⁠Extracts evenly spaced frames from videos for analysis.
•⁠  ⁠Utilizes the NVIDIA NEVA-22B API for action detection on individual frames.
•⁠  ⁠Compares detection success rates between two uploaded videos.
•⁠  ⁠Includes an interactive Gradio interface for user-friendly operation.

## Requirements

•⁠  ⁠Python 3.7 or above
•⁠  ⁠Required libraries:
  - ⁠ moviepy ⁠
  - ⁠ Pillow ⁠
  - ⁠ gradio ⁠
  - ⁠ requests ⁠

### Install all dependencies using the following command:
pip install moviepy pillow gradio requests


### Clone the repository to your local machine:
git clone https://github.com/your-repo-name/action-detection-neva22.git
cd action-detection-neva22

### Replace the placeholder API key in the script with your actual NVIDIA NEVA API key:
*API_KEY* = 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

### Run the script to start the Gradio interface:
python main.py


## Usage Instructions
 1. Launch the Interface
Execute the script to open the Gradio interface in your default web browser.

 2. Upload Videos
Video 1 (Synthetic): Upload a synthetic video file.
Video 2 (Real): Upload a real-world video file.

 3. Specify the Action
Enter the action you want the system to detect (e.g., "walking," "standing").

 4. Review Results
The system analyzes the videos and calculates the success rate for detecting the specified action in each video. Results are displayed in the interface.

## Functions

1.⁠ ⁠extract_frames_from_video(video_path, num_frames=16)
Extracts a specified number of evenly spaced frames from a video file.

2.⁠ ⁠encode_frame_to_base64(frame)
Converts a video frame (as a PIL Image) into a Base64-encoded string for API usage.

3.⁠ ⁠query_action_in_frame(frame_b64, action, retries=3)
Sends a Base64-encoded frame to the NVIDIA NEVA-22B API to identify the specified action.

4.⁠ ⁠calculate_detection_success_rate(frames, action, confidence_threshold=0.6)
Determines the success rate of detecting an action across multiple video frames based on a confidence threshold.

5.⁠ ⁠analyze_videos_for_action(video1_path, video2_path, action)
Processes two video files and calculates the success rates for detecting the specified action in both.

6.⁠ ⁠launch_interface()
Initializes and launches the Gradio-based interface for the project.

## API Usage
This project relies on the NVIDIA NEVA-22B API for detecting actions in video frames. Ensure that you have:

## A valid API key
Proper access permissions
For more details, refer to the [NVIDIA Developer Portal](https://developer.nvidia.com).

## Example 1:
Input:
Video 1: A synthetic video containing sitting actions.
Video 2: A real-world video with sitting actions.
Action: "sitting"
Output:
Video 1 'sitting' Detection Success Rate: 0.0%
Video 2 'sitting' Detection Success Rate: 81.25%
![example 1_ sitting_](https://github.com/user-attachments/assets/6525ca14-140f-4f5a-92e5-598b4d77380c)
![example 1_sitting](https://github.com/user-attachments/assets/aa5057dd-a9b8-42b3-8875-3a3297f88c76)



## Example 2:
Input:
Video 1: A synthetic video containing standing actions.
Video 2: A real-world video with standing actions.
Action: "standing"
Output:
Video 1 'standing' Detection Success Rate: 0.0%
Video 2 'standing' Detection Success Rate: 100.0%

![standing_example2_](https://github.com/user-attachments/assets/b6e9b91d-647b-49f0-bc66-1eb4e33eab1b)

![standing_example2](https://github.com/user-attachments/assets/b471e9a7-815d-42bd-b7d9-2a03a10ec91a)





![image](https://github.com/user-attachments/assets/a79b7f5f-750d-4c65-9008-a0420311bb8a)




### Demo Video
Check out the demo video here.


### Notes:
The NVIDIA NEVA-22B API requires an active internet connection for processing.
The system processes a maximum of 16 frames per video to optimize performance and accuracy.
Keep your API key private and avoid sharing it in public repositories.
vbnet

### Key Highlights:
•⁠  ⁠*Formatting*: Used appropriate Markdown syntax for headings, code blocks, images, and links.
•⁠  ⁠*Clarity*: The README is structured clearly to guide users from setup to usage, with a section for functions, API usage, and demo.
