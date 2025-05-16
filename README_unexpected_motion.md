# Object Movement Analysis with YOLO and Optical Flow

This project analyzes movement in a video using a combination of:
- **YOLOv3** for object detection
- **Lucas-Kanade Optical Flow** for motion tracking
- **Heatmaps and trajectory plots** for visualizing movement
- **Semantic reasoning** to detect unexpected motion (e.g., a "chair" that moves)

## Features

- Detects objects frame-by-frame using YOLOv3
- Tracks feature points using optical flow
- Builds a motion intensity heatmap over the video
- Plots individual motion trajectories for tracked points
- Flags "unexpected motion" based on semantic labels (e.g. static objects shouldn't move)

## Requirements

Install dependencies with:

```bash
pip install opencv-python numpy matplotlib
```

You'll also need:
- `yolov3.weights` – https://pjreddie.com/media/files/yolov3.weights
- `yolov3.cfg` – https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
- `coco.names` (or use built-in class list)

## Input

- Place your input video as `cube.mp4` (or modify the filename in the code).
- The script processes the video frame by frame.

## How It Works

1. **Detect objects** in each frame using YOLOv3.
2. **Track feature points** between frames with optical flow.
3. **Mark moving points**, draw motion vectors, and log trajectory.
4. **Build a heatmap** of all movement over time.
5. **Check object labels**:
   - If a **static object** (like `chair`, `tv`, `bed`) moves, it's flagged as unexpected.
6. **Visual outputs**:
   - A plotted trajectory of all moving points
   - A motion heatmap
   - An overlaid heatmap on the final video frame
   - Console logs for any anomalous motion

## "Unexpected Motion" Logic

```python
static_classes = {"chair", "tvmonitor", "bed", "sofa", "toilet"}
dynamic_classes = {"person", "car", "dog", "bicycle", "motorbike", ...}
```

If a static-class object contains more than `3` moving feature points inside its bounding box, it's flagged.

## Example Output

- `Moving Points Trajectory` – plots all tracked points over time
- `Motion Intensity Heatmap` – shows regions of high motion
- `Overlayed Heatmap` – superimposes the heatmap on a video frame
- Console output:
  ```
  Unexpected motion detected in 'chair' at (123, 220) with score: 7
  ```

## Notes

- Adjust frame resolution inside the script if your video isn't 640x480.
- You can increase `maxCorners` or tune `qualityLevel` for more/less optical flow points.
- The threshold for "unexpected motion" can be adjusted in the code.

## Sample Frame (if included)

_Add a screenshot or video overlay here if needed._



