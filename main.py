import requests
import base64
from moviepy.editor import VideoFileClip
from PIL import Image
from io import BytesIO
import gradio as gr
import time

INVOKE_URL = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
API_KEY = "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"  

def extract_frames_from_video(video_path, num_frames=16):
    """
    Extract evenly spaced frames from the video file.
    :param video_path: Path to the video file.
    :param num_frames: Number of frames to extract.
    :return: List of PIL Image frames.
    """
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        frames = [clip.get_frame(i * duration / num_frames) for i in range(num_frames)]
        return [Image.fromarray(frame) for frame in frames]
    except Exception as e:
        raise ValueError(f"Error extracting frames from video: {e}")

def encode_frame_to_base64(frame):
    """
    Convert a PIL image frame to a Base64 encoded string.
    :param frame: PIL Image frame.
    :return: Base64 encoded string.
    """
    buffered = BytesIO()
    frame.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def query_action_in_frame(frame_b64, action, retries=3):
    """
    Use NVIDIA NEVA API to detect if the specified action is performed in the image.
    :param frame_b64: Base64 encoded frame.
    :param action: Action to detect (e.g., "jumping", "running").
    :param retries: Number of retries in case of failure.
    :return: True if action detected, False otherwise.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'Do you see someone performing the action "{action}" in this image? <img src="data:image/png;base64,{frame_b64}" />',
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 0,
        "stream": False,
    }

    for _ in range(retries):
        try:
            response = requests.post(INVOKE_URL, headers=headers, json=payload)
            response.raise_for_status()  
            result = response.json()
            return "yes" in result.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
        except requests.RequestException as e:
            print(f"Error querying the API: {e}")
            time.sleep(2)  
    return False  

def calculate_detection_success_rate(frames, action, confidence_threshold=0.6):
    """
    Calculate the success rate of detecting the specified action in the frames.
    :param frames: List of PIL frames.
    :param action: Action to detect (e.g., "jumping", "running").
    :return: Detection success rate as a percentage.
    """
    successful_detections = 0
    total_frames = len(frames)
    
    for frame in frames:
        frame_b64 = encode_frame_to_base64(frame)
        if query_action_in_frame(frame_b64, action):
            successful_detections += 1

    return (successful_detections / total_frames) * 100

def analyze_videos_for_action(video1_path, video2_path, action):
    """
    Process two videos and calculate action detection success rates for each.
    :param video1_path: Path to the first video (e.g., synthetic).
    :param video2_path: Path to the second video (e.g., real).
    :param action: The action to detect (e.g., "jumping", "running").
    :return: Analysis result strings for both videos.
    """
    try:
        frames1 = extract_frames_from_video(video1_path)
        frames2 = extract_frames_from_video(video2_path)

        success_rate1 = calculate_detection_success_rate(frames1, action)
        success_rate2 = calculate_detection_success_rate(frames2, action)

        return (
            f"Video 1 '{action}' Detection Success Rate: {success_rate1:.2f}%",
            f"Video 2 '{action}' Detection Success Rate: {success_rate2:.2f}%",
        )
    except Exception as e:
        return f"Error processing videos: {str(e)}", None

def launch_interface():
    """Launch the Gradio interface for video action detection."""
    
    video_input1 = gr.Video(label="Upload Synthetic Video (Video 1)")
    video_input2 = gr.Video(label="Upload Real Video (Video 2)")
    action_input = gr.Textbox(
        label="Action to Detect (e.g., jumping, running)",
        placeholder="Enter the action you want to detect in the videos"
    )
    

    output_video1 = gr.Textbox(label="Analysis of Video 1 (Synthetic)", placeholder="Detection results for Video 1 will appear here.", lines=2)
    output_video2 = gr.Textbox(label="Analysis of Video 2 (Real)", placeholder="Detection results for Video 2 will appear here.", lines=2)
    

    interface = gr.Interface(
        fn=analyze_videos_for_action,
        inputs=[video_input1, video_input2, action_input],
        outputs=[output_video1, output_video2],
        title="Action Detection with NVIDIA NEVA-22B",
        description=(
            "Upload two videos (one synthetic and one real) and specify an action to detect. "
            "The system will analyze the videos and calculate the success rate for detecting the specified action in each video."
        ),
        allow_flagging="never",  
        live=True,  
    )
    
    interface.launch(share=True)

if __name__ == "__main__":
    launch_interface()
