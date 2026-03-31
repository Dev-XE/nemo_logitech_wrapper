# nemo_logitech_wrapper

A ROS2 Python package for streaming video from a Logitech C270 HD webcam and performing real-time ArUco marker detection and pipeline inspection. Designed for autonomous underwater vehicle (AUV) pipeline inspection missions.

### Running Both on Same Machine (Testing)

```bash
# Terminal 1: Start receiver
ros2 run nemo_logitech_wrapper receiver --ros-args -p port:=5001
```

```bash
# Terminal 2: Start transmitter(jetson)
ros2 run nemo_logitech_wrapper transmitter --ros-args -p device:=/dev/video3
```

## Features

- **Video Streaming**: H.264 encoded RTP/UDP streaming with configurable parameters
- **ArUco Detection**: Multi-dictionary support (4x4, 5x5, 6x6, 7x7) with parallel detection threads
- **Pipeline Detection**: HSV-based pipeline contour detection with orientation tracking
- **Professional HUD**: Real-time telemetry overlay with mission state tracking
- **Day/Night Mode**: Automatic brightness detection with CLAHE enhancement

## Prerequisites

### System Dependencies

```bash
# GStreamer
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav

# Python GStreamer bindings
sudo apt install python3-gi gir1.2-gstreamer-1.0

# OpenCV with GStreamer support
sudo apt install python3-opencv
```

### ROS2 Setup

Ensure you have a ROS2 workspace set up (tested with Humble/Iron).

## Installation

```bash
# Navigate to your ROS2 workspace src directory
cd ~/ros2_ws/src

# Clone the repository
git clone https://github.com/your-username/nemo_logitech_wrapper.git

# Build the package
cd ..
colcon build --packages-select nemo_logitech_wrapper

# Source the workspace
source install/setup.bash
```

## Usage

### Transmitter (Logitech Webcam Streaming)

Stream video from the Logitech C270 webcam to a remote host:

```bash
ros2 run nemo_logitech_wrapper transmitter
```

**Default Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `/dev/video2` | V4L2 device path |
| `host` | `192.168.2.1` | Destination IP address |
| `port` | `5001` | UDP port |
| `width` | `640` | Video width |
| `height` | `480` | Video height |
| `fps` | `30` | Frame rate |
| `bitrate_kbps` | `5000` | H.264 bitrate |
| `speed_preset` | `veryfast` | x264 encoding speed |
| `key_int_max` | `60` | Keyframe interval |
| `use_videoconvert` | `true` | Use videoconvert element |

**Example with custom parameters:**

```bash
ros2 run nemo_logitech_wrapper transmitter --ros-args \
    -p device:=/dev/video0 \
    -p host:=192.168.1.100 \
    -p port:=5001 \
    -p width:=1280 \
    -p height:=720 \
    -p fps:=30 \
    -p bitrate_kbps:=8000
```

### Receiver (ArUco & Pipeline Detection)

Receive the stream and run detection:

```bash
ros2 run nemo_logitech_wrapper receiver
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `port` | `5001` | UDP port to listen on |
| `latency_ms` | `10` | RTP jitter buffer latency |
| `sync` | `false` | Synchronize with sender |

```bash
ros2 run nemo_logitech_wrapper receiver --ros-args \
    -p port:=5001 \
    -p latency_ms:=20
```

### Running Both on Same Machine (Testing)

```bash
# Terminal 1: Start receiver
ros2 run nemo_logitech_wrapper receiver --ros-args -p port:=5001

# Terminal 2: Start transmitter (stream to localhost)
ros2 run nemo_logitech_wrapper transmitter --ros-args \
    -p host:=127.0.0.1 \
    -p port:=5001
```

## ArUco Marker Configuration

The receiver detects specific ArUco markers for pipeline inspection:

| ArUco ID | Role | Description |
|----------|------|-------------|
| 56 | PIPELINE ENTRY | Pipeline detected - move to next section |
| 5 | SECTION 2 | Section 2 reached - proceed forward |
| 20 | SECTION 3 | Section 3 reached - scan surroundings |
| 32 | PIPELINE EXIT | Pipeline exit - inspection complete |
| Others | DOCKING | Docking marker for alignment |

## Pipeline Detection

The system detects underwater pipelines using HSV color filtering:
- Orange/yellow pipeline detection
- Contour validation with aspect ratio, fill ratio, and solidity checks
- Orientation angle calculation for alignment

## Display Windows

- **Pipeline Inspection Feed**: Main video with HUD overlay
- **Pipeline Mask**: Binary mask showing detected pipeline contours

Press `Q` to quit the receiver.

## Architecture

```
[Logitech C270] → v4l2src → x264enc → rtph264pay → udpsink
                                                        ↓
                                              [Remote Host:port]
                                                        ↓
[Display] ← appsink ← videoconvert ← avdec_h264 ← rtph264depay ← udpsrc
```

## Troubleshooting

### GStreamer Pipeline Issues

```bash
# Check if device exists
ls -la /dev/video*

# Test GStreamer pipeline manually
gst-launch-1.0 v4l2src device=/dev/video2 ! video/x-raw,width=640,height=480,framerate=30/1 ! \
    x264enc tune=zerolatency ! fakesink
```

### OpenCV GStreamer Support

Verify OpenCV has GStreamer support:

```python
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
```

If GStreamer is not listed, reinstall OpenCV with GStreamer support.

