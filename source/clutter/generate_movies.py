"""Generates videos of moving MNIST characters with configurable parameters.
Each video contains one foreground character moving at a constant speed and 
multiple background characters moving with random walk dynamics. The videos 
are saved in MP4 format, and a corresponding TSV file logs the positions and 
identities of the characters in each frame.
The MNIST dataset is loaded using PyTorch/torchvision to avoid issues with 
deprecated libraries.

Initial code was produced with Gemini AI. """

import os
import numpy as np
import numpy.lib.format as npfmt

import argparse
import csv
from dataclasses import dataclass, field
from typing import List, Literal, Tuple
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Configuration ---

@dataclass

class StimulusConfig:
    """Holds all configuration parameters for stimulus generation."""
    # Video Properties
    width: int = 128
    height: int = 128
    duration_seconds: int = 30
    fps: int = 30

    # Foreground Character Properties
    fg_speeds: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])

    # Background Character Properties
    bg_char_counts: List[int] = field(default_factory=lambda: [5, 10, 15])
    bg_mean_speeds: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])

    # Switch Event Properties
    mean_switch_interval_seconds: float = 5.0
    switch_mode: Literal["exclusive", "joint"] = "exclusive"

    # Output Settings
    output_dir: str = "stimulus_output"
    num_videos: int = 5
    # "full": mp4 + stimulus npy/tsv + mnist_images/mnist_labels npy;
    # "simple": stimulus npy and tsv only.
    output_mode: Literal["full", "simple"] = "full"

    # MNIST sample range (by index, not digit)
    mnist_sample_start: int = 0
    mnist_sample_end: int = 60000  # default: all samples

    suffix: str = ""
    storage_dtype: Literal["uint8", "float32"] = "uint8"


class MovingCharacter:
    """Represents a single moving MNIST character."""
    def __init__(self, label: int, image: np.ndarray, pos: np.ndarray, vel: np.ndarray):
        self.label = label
        self.image = image
        self.height, self.width = image.shape
        # pos is the center of the character
        self.pos = pos.astype(float)
        self.vel = vel.astype(float)
        self.center_x = float(pos[0])
        self.center_y = float(pos[1])

    def update_position(self, frame_dims: Tuple[int, int]):
        """Updates the character's position (center) and handles bouncing off edges."""
        frame_h, frame_w = frame_dims
        self.pos += self.vel

        # Bounce off horizontal walls (left/right)
        left = self.pos[0] - self.width / 2
        right = self.pos[0] + self.width / 2
        if left < 0:
            self.pos[0] = self.width / 2
            self.vel[0] *= -1
        elif right > frame_w:
            self.pos[0] = frame_w - self.width / 2
            self.vel[0] *= -1

        # Bounce off vertical walls (top/bottom)
        top = self.pos[1] - self.height / 2
        bottom = self.pos[1] + self.height / 2
        if top < 0:
            self.pos[1] = self.height / 2
            self.vel[1] *= -1
        elif bottom > frame_h:
            self.pos[1] = frame_h - self.height / 2
            self.vel[1] *= -1
            
    def update_random_walk(self, frame_dims: Tuple[int, int], mean_speed: float):
        """Updates velocity with a random component and then updates position."""
        # Add a small random perturbation to the velocity
        # perturbation = (np.random.rand(2) - 0.5) * mean_speed
        # self.vel += perturbation
        # # Normalize to keep speed around the mean, but allow fluctuations
        # current_speed = np.linalg.norm(self.vel)
        # if current_speed > 0:
        #     self.vel = self.vel / current_speed * np.random.normal(loc=mean_speed, scale=mean_speed/2)
        # # Ensure minimum movement
        # if np.linalg.norm(self.vel) < 0.1:
        #     angle = np.random.uniform(0, 2 * np.pi)
        #     self.vel = np.array([np.cos(angle), np.sin(angle)]) * mean_speed

        angle = np.random.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)]) * mean_speed
        # This call ensures bouncing logic is applied
        self.update_position(frame_dims)

def load_mnist_data(config=None):
    """Loads MNIST dataset using PyTorch/torchvision and organizes it by digit. Optionally restricts to a sample index range."""
    import torchvision

    print("Loading MNIST dataset using PyTorch/torchvision...")
    mnist_dataset = torchvision.datasets.MNIST(
        root='./mnist_data_pytorch',
        train=True,
        download=True
    )
    sample_start = config.mnist_sample_start if config is not None and hasattr(config, 'mnist_sample_start') else 0
    sample_end = config.mnist_sample_end if config is not None and hasattr(config, 'mnist_sample_end') else len(mnist_dataset)
    sample_end = min(sample_end, len(mnist_dataset))
    # Select only the specified range
    selected_images = []
    selected_labels = []
    mnist_digits = {i: [] for i in range(10)}
    for idx in range(sample_start, sample_end):
        image, label = mnist_dataset[idx]
        arr = np.array(image)
        mnist_digits[label].append(arr)
        selected_images.append(arr)
        selected_labels.append(label)
    if config is not None and getattr(config, "output_mode", "full") == "full":
        np.save(
            os.path.join(config.output_dir, f"mnist_images_{config.suffix}.npy"),
            np.stack(selected_images),
        )
        np.save(
            os.path.join(config.output_dir, f"mnist_labels_{config.suffix}.npy"),
            np.array(selected_labels),
        )
    print(f"MNIST data loaded and processed. Samples used: {sample_start}-{sample_end-1}")
    return mnist_digits

def get_random_digit(mnist_data: dict, digit: int = None) -> Tuple[int, np.ndarray]:
    """Selects a random image for a given digit, or a random digit."""
    if digit is None:
        digit = np.random.randint(0, 10)
    
    images_for_digit = mnist_data[digit]
    image = images_for_digit[np.random.randint(0, len(images_for_digit))]
    return digit, image

def paste_character(frame: np.ndarray, char: MovingCharacter):
    """Pastes a character's image onto the frame."""
    h, w = char.image.shape
    # Calculate top-left corner from center position
    x = int(round(char.pos[0] - w / 2))
    y = int(round(char.pos[1] - h / 2))

    y_start, y_end = max(0, y), min(frame.shape[0], y + h)
    x_start, x_end = max(0, x), min(frame.shape[1], x + w)

    img_y_start = max(0, -y)
    img_x_start = max(0, -x)
    img_y_end = h - max(0, (y + h) - frame.shape[0])
    img_x_end = w - max(0, (x + w) - frame.shape[1])

    char_roi = char.image[img_y_start:img_y_end, img_x_start:img_x_end]

    if char_roi.size == 0:
        return

    # Update center_x and center_y to the actual center of the pasted region
    char.center_x = (x_start + x_end) / 2
    char.center_y = (y_start + y_end) / 2

    frame_roi = frame[y_start:y_end, x_start:x_end]
    if frame.dtype == np.uint8:
        # Widen before addition so overlapping bright pixels saturate instead of wrapping.
        combined = frame_roi.astype(np.uint16) + char_roi.astype(np.uint16)
        frame[y_start:y_end, x_start:x_end] = np.clip(combined, 0, 255).astype(np.uint8)
    else:
        combined = frame_roi.astype(np.float32) + char_roi.astype(np.float32)
        frame[y_start:y_end, x_start:x_end] = np.clip(combined, 0.0, 255.0).astype(
            np.float32
        )


# --- MODIFIED FUNCTION ---
def generate_stimulus_video(config: StimulusConfig, mnist_data: dict):
    """Generates stimulus npy + tsv; optionally mp4 when output_mode is full."""
    
    # --- Initialization ---
    suffix = config.suffix
    write_mp4 = config.output_mode == "full"
    video_filename = f"stimulus_{suffix}.mp4"
    tsv_filename = f"stimulus_{suffix}.tsv"
    video_path = os.path.join(config.output_dir, video_filename)
    tsv_path = os.path.join(config.output_dir, tsv_filename)
    npy_path = os.path.join(config.output_dir, f"stimulus_{suffix}.npy")

    video_writer = None
    if write_mp4:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            video_path, fourcc, config.fps, (config.width, config.height)
        )

    tsv_file = open(tsv_path, 'w', newline='')
    tsv_writer = csv.writer(tsv_file, delimiter='\t')
    # Add new columns to the header
    tsv_writer.writerow([
        'frame', 'fg_char_id', 'fg_char_x', 'fg_char_y', 
        'bg_char_ids', 'fg_speed', 'bg_mean_speed',
        'fg_switch', 'bg_switch'
    ])

    total_frames = config.duration_seconds * config.fps
    mean_switch_interval_frames = config.mean_switch_interval_seconds * config.fps
    frame_dims = (config.height, config.width)

    storage_dtype = np.dtype(config.storage_dtype)
    npy_data = npfmt.open_memmap(
        npy_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(total_frames, config.height, config.width),
    )

    # --- Initial State Setup ---
    fg_speed = np.random.choice(config.fg_speeds)
    angle = np.random.uniform(0, 2 * np.pi)
    fg_vel = np.array([np.cos(angle), np.sin(angle)]) * fg_speed
    fg_label, fg_img = get_random_digit(mnist_data)
    # Center position: ensure the full character fits in the frame
    fg_pos = np.random.rand(2) * [config.width - 28, config.height - 28] + 14
    fg_char = MovingCharacter(fg_label, fg_img, fg_pos, fg_vel)

    bg_char_count = np.random.choice(config.bg_char_counts)
    bg_mean_speed = np.random.choice(config.bg_mean_speeds)
    background_chars = []
    for _ in range(bg_char_count):
        bg_label, bg_img = get_random_digit(mnist_data)
        bg_pos = np.random.rand(2) * [config.width - 28, config.height - 28] + 14
        bg_vel = (np.random.rand(2) - 0.5) * 2 * bg_mean_speed
        background_chars.append(MovingCharacter(bg_label, bg_img, bg_pos, bg_vel))

    next_switch_frame = int(np.random.exponential(scale=mean_switch_interval_frames))

    def change_foreground() -> None:
        """Resample the foreground identity, position, and speed."""
        nonlocal fg_speed
        fg_speed = np.random.choice(config.fg_speeds)
        fg_label, fg_img = get_random_digit(mnist_data)
        if np.linalg.norm(fg_char.vel) == 0:
            current_direction = np.array([1.0, 0.0])
        else:
            current_direction = fg_char.vel / np.linalg.norm(fg_char.vel)

        fg_char.vel = current_direction * fg_speed
        fg_char.pos = np.random.rand(2) * [config.width - 28, config.height - 28] + 14
        fg_char.label = fg_label
        fg_char.image = fg_img

    def change_background() -> None:
        """Resample all background characters and their shared speed scale."""
        nonlocal bg_char_count, bg_mean_speed, background_chars
        bg_char_count = np.random.choice(config.bg_char_counts)
        bg_mean_speed = np.random.choice(config.bg_mean_speeds)
        background_chars = []
        for _ in range(bg_char_count):
            bg_label, bg_img = get_random_digit(mnist_data)
            bg_pos = np.random.rand(2) * [config.width - 28, config.height - 28] + 14
            angle = np.random.uniform(0, 2 * np.pi)
            bg_vel = np.array([np.cos(angle), np.sin(angle)]) * bg_mean_speed
            background_chars.append(MovingCharacter(bg_label, bg_img, bg_pos, bg_vel))

    # --- Simulation Loop ---
    for frame_idx in tqdm(range(total_frames), desc=f"Generating Video"):
        # Initialize switch flags for the current frame
        fg_switch_flag = 0
        bg_switch_flag = 0

        # Check for switch event
        if frame_idx >= next_switch_frame:
            if config.switch_mode == "joint":
                # --- Change Foreground and Background together ---
                fg_switch_flag = 1
                bg_switch_flag = 1
                change_foreground()
                change_background()
            elif config.switch_mode == "exclusive" and np.random.rand() < 0.5:
                # --- Change Foreground ---
                fg_switch_flag = 1  # Set flag to 1 for this frame
                change_foreground()
            elif config.switch_mode == "exclusive":
                # --- Change Background ---
                bg_switch_flag = 1  # Set flag to 1 for this frame
                change_background()
            else:
                raise ValueError(f"Unknown switch_mode: {config.switch_mode}")

            next_switch_frame = frame_idx + int(np.random.exponential(scale=mean_switch_interval_frames))

        # --- Update Positions ---
        fg_char.update_position(frame_dims)
        for char in background_chars:
            char.update_random_walk(frame_dims, bg_mean_speed)

        # --- Render Frame ---
        frame = np.zeros(frame_dims, dtype=storage_dtype)
        for char in background_chars:
            paste_character(frame, char)
        paste_character(frame, fg_char)

        npy_data[frame_idx] = frame

        # --- Log Data ---
        bg_char_id_list = [str(c.label) for c in background_chars]
        bg_char_ids_str = ",".join(bg_char_id_list)
        # Write the row with the new switch flags
        tsv_writer.writerow([
            frame_idx,
            fg_char.label,
            f"{fg_char.center_x:.2f}",
            f"{fg_char.center_y:.2f}",
            bg_char_ids_str,
            f"{np.linalg.norm(fg_char.vel):.2f}",
            f"{bg_mean_speed:.2f}",
            fg_switch_flag,
            bg_switch_flag
        ])

        if video_writer is not None:
            video_writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR))

    # --- Cleanup ---
    if video_writer is not None:
        video_writer.release()
    tsv_file.close()
    if write_mp4:
        print(f"Successfully generated {video_path}, {tsv_path}, and {npy_path}")
    else:
        print(f"Successfully generated {tsv_path}, and {npy_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate moving MNIST stimulus sequences.")
    p.add_argument(
        "--output-mode",
        choices=("full", "simple"),
        default="simple",
        help="full: mp4 + stimulus npy/tsv + mnist npy; simple: stimulus npy + tsv only.",
    )
    p.add_argument(
        "--hour",
        type=int,
        default=10,
        metavar="H",
        help="Hours of train stimulus (suffix becomes Hh-<dtype>; train duration = 3600*H seconds).",
    )
    p.add_argument(
        "--storage-dtype",
        choices=("uint8", "float32"),
        default="uint8",
        help="On-disk stimulus dtype. Default uint8 minimizes shared-filesystem I/O.",
    )
    p.add_argument(
        "--split",
        choices=("all", "train", "validation", "test"),
        default="all",
        help="Which split to generate. Default all preserves the historical behavior.",
    )
    p.add_argument(
        "--switch-mode",
        choices=("exclusive", "joint"),
        default="exclusive",
        help="exclusive: each switch changes fg or bg; joint: each switch changes both.",
    )
    p.add_argument(
        "--suffix-extra",
        type=str,
        default=None,
        help=(
            "Extra text appended to the Hh-<dtype> suffix. "
            "Default is '-jointswitch' for --switch-mode joint and empty otherwise."
        ),
    )
    return p.parse_args()


def main():
    """Main function to orchestrate stimulus generation."""
    args = parse_args()
    ## Normal data
    data_hour_length = args.hour
    suffix_extra = args.suffix_extra
    if suffix_extra is None:
        suffix_extra = "-jointswitch" if args.switch_mode == "joint" else ""
    data_suffix = f"{data_hour_length}h-{args.storage_dtype}{suffix_extra}"
    config = StimulusConfig(
        width=96,
        height=96,
        duration_seconds=3600 * data_hour_length,
        fps=24,
        fg_speeds=[1,0, 2.0, 3.0, 4.0, 6.0, 8.0], #[1.0, 2.0, 4.0],
        bg_char_counts=[1, 2, 4, 8, 12], #[1, 2, 4],
        bg_mean_speeds=[1.0, 2.0, 4.0, 6.0, 8.0], #[1.0, 2.0, 4.0],
        mean_switch_interval_seconds=1.0,
        switch_mode=args.switch_mode,
        output_dir=os.path.join(PROJECT_ROOT, "stimuli"),
        mnist_sample_start=0,
        mnist_sample_end=40000,
        suffix="reg-train-" + data_suffix,
        output_mode=args.output_mode,
        storage_dtype=args.storage_dtype,
    )
    os.makedirs(config.output_dir, exist_ok=True)

    def run_split(
        split_name: Literal["train", "validation", "test"],
        sample_start: int,
        sample_end: int,
        duration_seconds: int,
    ) -> None:
        config.mnist_sample_start = sample_start
        config.mnist_sample_end = sample_end
        config.duration_seconds = duration_seconds
        config.suffix = f"reg-{split_name}-{data_suffix}"
        mnist_data = load_mnist_data(config)
        print("MNIST data loaded.")
        generate_stimulus_video(config, mnist_data)

    if args.split in ("all", "train"):
        run_split("train", 0, 40000, 3600 * data_hour_length)

    if args.split in ("all", "validation"):
        run_split("validation", 40000, 50000, 2400)

    if args.split in ("all", "test"):
        run_split("test", 50000, 60000, 2400)

if __name__ == "__main__":
    main()
