import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve paths relative to project root (utils_viz is under project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the npy file
npy_file = _PROJECT_ROOT / "stimuli" / "stimulus_1bg-test.npy"
data = np.load(str(npy_file), allow_pickle=True)

# Print information about the data
print(f"Data shape: {data.shape}")
print(f"Data dtype: {data.dtype}")
print(f"Data min: {data.min()}, max: {data.max()}")

# Determine the shape and extract first 10 frames
if len(data.shape) == 3:
    # Shape: (num_frames, height, width) or (num_videos, num_frames, height, width)
    if data.shape[0] > 10:
        # Assume first dimension is frames
        frames = data[:10]
    else:
        # If less than 10 frames, take all
        frames = data
        print(f"Warning: Only {data.shape[0]} frames available, using all")
elif len(data.shape) == 4:
    # Shape: (num_videos, num_frames, height, width)
    # Take first video and first 10 frames
    num_frames_to_show = min(10, data.shape[1])
    frames = data[0, :num_frames_to_show]
    print(f"Using first video, showing {num_frames_to_show} frames")
else:
    raise ValueError(f"Unexpected data shape: {data.shape}")

print(f"\nExtracted frames shape: {frames.shape}")
print(f"Number of frames to display: {len(frames)}")

# Print information about each frame
for i in range(len(frames)):
    print(f"Frame {i}: shape={frames[i].shape}, min={frames[i].min():.2f}, max={frames[i].max():.2f}")

# Create a figure with 10 subplots in a row
fig, axes = plt.subplots(1, len(frames), figsize=(2*len(frames), 2))

# If only one frame, make axes iterable
if len(frames) == 1:
    axes = [axes]

# Display each frame
for i, frame in enumerate(frames):
    ax = axes[i]
    # Normalize frame to 0-1 range if needed
    if frame.max() > 1.0:
        frame_display = frame / 255.0
    else:
        frame_display = frame
    
    ax.imshow(frame_display, cmap='gray')
    ax.set_title(f'Frame {i}')
    ax.axis('off')

plt.tight_layout()

# Save the figure
output_file = _PROJECT_ROOT / "first_10_frames.pdf"
plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: {output_file}")

plt.close()
print("Done!")

