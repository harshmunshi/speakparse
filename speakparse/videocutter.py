import cv2
import face_alignment
import os
import numpy as np
import math

# lm 62, 66

# initialize the face aligner
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu")

# load the video
video_path = os.path.join(os.getcwd(), "tests", "assets", "videos", "paul2_src.mp4")

cap = cv2.VideoCapture(video_path)

# make buckets of individual frames
speaking = []
silent = []

# function loop to convert frame list to video
import cv2

def create_video_from_frames(frame_list, output_file, fps=30, frame_size=None):
    """
    Create a video from a list of frames.

    :param frame_list: List of frames (numpy arrays)
    :param output_file: Output video file path
    :param fps: Frames per second
    :param frame_size: Size of the frame (width, height). If None, use the size of the first frame.
    """
    if not frame_list:
        raise ValueError("Frame list is empty")

    # Get frame size from the first frame if not provided
    if frame_size is None:
        frame_size = (frame_list[0].shape[1], frame_list[0].shape[0])

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    for frame in frame_list:
        # Ensure frame is the correct size
        if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
            frame = cv2.resize(frame, frame_size)
        
        out.write(frame)

    out.release()
    print(f"Video saved as {output_file}")

def visualize_landmarks(image, landmarks):
    """
    Visualize landmarks on an image.

    :param image: Input image (numpy array)
    :param landmarks: A NumPy array of shape (n, 2) representing landmarks (x, y) coordinates
    """
    if not isinstance(landmarks, np.ndarray) or landmarks.shape[1] != 2:
        raise ValueError("Landmarks should be a NumPy array of shape (n, 2)")
    print(math.dist(landmarks[62], landmarks[66]))
    for idx, (x, y) in enumerate(landmarks):
        if idx == 62 or idx == 66:
          # Draw the landmark point
          cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
        
          # Annotate the landmark with its index
          cv2.putText(image, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return image

while True:
	ret, frame = cap.read()
	if not ret:
		break
	# BGR -> RGB
	preds = fa.get_landmarks(frame[..., ::-1])
	print(preds)
	# im = visualize_landmarks(frame, preds[0])
	# cv2.imwrite("sample.png", im)
	if math.dist(preds[0][62], preds[0][66]) < 4:
		silent.append(frame)
	else:
		speaking.append(frame)

create_video_from_frames(silent, "silent.avi")
create_video_from_frames(speaking, "speaking.avi")
