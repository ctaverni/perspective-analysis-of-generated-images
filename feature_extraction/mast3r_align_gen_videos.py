import argparse
import copy
import functools
import math
import os
import shutil
import tempfile
from contextlib import nullcontext

import cv2
import gradio
import mast3r.utils.path_to_dust3r  # noqa
import matplotlib.pyplot as pl
import numpy as np
import PIL
import torch
import torchvision.transforms as tvf
import trimesh
from dust3r.demo import get_args_parser as dust3r_get_args_parser
from dust3r.demo import set_print_with_timestamp
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images
from dust3r.viz import CAM_COLORS, OPENGL, add_scene_cam, cat_meshes, pts3d_to_trimesh
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.demo import get_args_parser, main_demo
from mast3r.image_pairs import make_pairs
from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.processor import Retriever
from mast3r.utils.misc import hash_md5
from PIL import Image
from PIL.ImageOps import exif_transpose
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def load_video_as_frames(video_path, load_frequency=1, max_n_frames=np.inf):
    """
    Load an RGB video as a set of frames.

    Args:
        video_path (str): Path to the video file.
        size (tuple): Desired size of the frames (width, height).
        verbose (bool): Whether to print progress.

    Returns:
        list: A list of frames (each frame is a numpy array).
    """
    frames = []
    files = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % load_frequency != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        # # Resize frame to the desired size
        # frame = cv2.resize(frame, size)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(frame)

        img = exif_transpose(pil_image).convert("RGB")

        W1, H1 = img.size

        img = _resize_pil_image(img, round(224 * max(W1 / H1, H1 / W1)))

        W, H = img.size
        cx, cy = W // 2, H // 2
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))

        W2, H2 = img.size

        frames.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(frames),
                instance=str(len(frames)),
            )
        )

        fname = f"{frame_idx:04d}.jpg"
        os.makedirs("input_data", exist_ok=True)
        pil_image.save(os.path.join("input_data", fname))
        files.append(fname)

        if len(frames) >= max_n_frames:
            break

    cap.release()
    return frames, files


def align_images(
    video_path,
    model,
    scene_graph="swin-1-noncyclic",
    device="cuda",
    img_freq=1,
    max_n_frames=np.inf,
    outdir="./outputs",
):
    imgs, filelist = load_video_as_frames(
        video_path, load_frequency=img_freq, max_n_frames=max_n_frames
    )

    pairs = make_pairs(
        imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=None
    )

    scene = sparse_global_alignment(filelist, pairs, outdir, model, device=device)

    return scene


def get_matches(sparse_ga, idx1, idx2, outdir="./outputs"):
    img1_fname, img2_fname = sparse_ga.img_paths[idx1], sparse_ga.img_paths[idx2]
    img_1_hash, img_2_hash = hash_md5(img1_fname), hash_md5(img2_fname)
    img_1_shape, img_2_shape = (
        sparse_ga.imgs[idx1].shape[:2],
        sparse_ga.imgs[idx2].shape[:2],
    )

    corres_conf_fwd = os.path.join(
        outdir, f"corres_conf=desc_conf_subsample=8/{img_1_hash}-{img_2_hash}.pth"
    )
    score, (xy1, xy2, confs) = torch.load(corres_conf_fwd)

    H0, W0 = img_1_shape
    H1, W1 = img_2_shape

    valid_matches_im0 = (
        (xy1[:, 0] >= 3)
        & (xy1[:, 0] < int(W0) - 3)
        & (xy1[:, 1] >= 3)
        & (xy1[:, 1] < int(H0) - 3)
    )

    valid_matches_im1 = (
        (xy2[:, 0] >= 3)
        & (xy2[:, 0] < int(W1) - 3)
        & (xy2[:, 1] >= 3)
        & (xy2[:, 1] < int(H1) - 3)
    )

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = xy1[valid_matches], xy2[valid_matches]
    confs = confs[valid_matches]

    matches_im0 = matches_im0.cpu().numpy()
    matches_im1 = matches_im1.cpu().numpy()
    confs = confs.cpu().numpy()

    return matches_im0, matches_im1, confs


def get_matches_clouds(sparse_ga, idx1, idx2, dense_points=None, outdir="./outputs"):
    matches_im0, matches_im1, _ = get_matches(sparse_ga, idx1, idx2, outdir=outdir)

    if dense_points is None:
        dense_points = sparse_ga.get_dense_pts3d()

    im_1_points = dense_points[0][idx1].reshape(224, 224, 3)
    im_2_points = dense_points[0][idx2].reshape(224, 224, 3)

    # Using the keypoints, get the corresponding points in each image
    # This seems to be the correct ordering (makes sense as matches will be row, col (y, x) and points will be (x,y))
    im_1_keypoint_cloud = im_1_points[matches_im0[:, 1], matches_im0[:, 0]]
    im_2_keypoint_cloud = im_2_points[matches_im1[:, 1], matches_im1[:, 0]]

    return im_1_keypoint_cloud, im_2_keypoint_cloud


def get_match_cloud_metrics(im_1_keypoint_cloud, im_2_keypoint_cloud):
    distances = torch.linalg.norm(
        im_1_keypoint_cloud - im_2_keypoint_cloud, dim=1, ord=2
    )

    mean = distances.mean().item()
    std = distances.std().item()
    max = distances.max().item()
    min = distances.min().item()
    median = distances.median().item()

    total_dist = distances.sum().item()
    n_points = distances.shape[0]

    mse = torch.nn.functional.mse_loss(im_1_keypoint_cloud, im_2_keypoint_cloud).item()

    output = {
        "mean": mean,
        "std": std,
        "max": max,
        "min": min,
        "median": median,
        "total_dist": total_dist,
        "n_points": n_points,
        "mse": mse,
    }

    return output


def windowed_align(sparse_ga, window_size=1, outdir="./outputs"):

    n_frames = len(sparse_ga.img_paths)
    assert window_size > 0

    dense_points = sparse_ga.get_dense_pts3d()

    # For this we will only consider forward matches. This could be done with reverse matches too,
    # but I think they should be the same? So it would be a waste of time.
    mean_distances = []
    metrics_all = []
    for i in tqdm(range(n_frames - 1)):
        frame_mean_distances = []
        frame_metrics = []
        for j in tqdm(
            range(i + 1, min(i + window_size + 1, n_frames)), position=1, leave=False
        ):
            im_1_keypoint_cloud, im_2_keypoint_cloud = get_matches_clouds(
                sparse_ga, i, j, dense_points=dense_points, outdir=outdir
            )
            if im_1_keypoint_cloud.shape[0] == 0 or im_2_keypoint_cloud.shape[0] == 0:
                frame_mean_distances.append(None)
                continue
            metrics = get_match_cloud_metrics(im_1_keypoint_cloud, im_2_keypoint_cloud)

            frame_mean_distances.append(metrics["mean"])
            frame_metrics.append(metrics)

        mean_distances.append(frame_mean_distances)
        metrics_all.append(frame_metrics)

    return mean_distances, metrics_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_graph", type=str, default="swin-10-noncyclic")
    # Scannet++ is 60fps so this is 10fps
    parser.add_argument("--img_freq", type=int, default=2)
    # This is 6 seconds of video then
    parser.add_argument("--max_n_frames", type=int, default=60)
    parser.add_argument("--align_window_size", type=int, default=10)
    parser.add_argument(
        "--outdir", type=str, default="/scratch/toskov/data/geom_align_outputs_gen/"
    )
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--end_sample", type=int, default=1e7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda"

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    gen_data_dir = "/scratch/toskov/geometry_consistency/gen_videos/"
    gen_files = os.listdir(gen_data_dir)

    output_root = args.outdir
    os.makedirs(output_root, exist_ok=True)

    for file in gen_files[args.start_sample : args.end_sample]:
        vid_path = os.path.join(gen_data_dir, file)
        if not os.path.exists(vid_path):
            print(f"Video {vid_path} does not exist")
            continue

        print(f"Processing {file}")

        outdir = os.path.join(output_root, file.replace(".mp4", ""))
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        scene = align_images(
            vid_path,
            model,
            scene_graph=args.scene_graph,
            device=device,
            img_freq=args.img_freq,
            max_n_frames=args.max_n_frames,
            outdir=outdir,
        )

        mean_dists, metrics = windowed_align(
            scene, window_size=args.align_window_size, outdir=outdir
        )

        mean_array = np.array(
            [np.mean(np.array(x)[np.array(x) != None]) for x in mean_dists]
        )

        # plot mean distance per frame through time
        pl.plot(mean_array)
        pl.xlabel("Frame")
        pl.ylabel("Mean distance")
        pl.grid()
        pl.savefig(os.path.join(outdir, "mean_distance.png"))
        pl.close()

        print("Mean error:", mean_array.mean())

        # Dump metrics
        metrics_path = os.path.join(outdir, "metrics.pth")
        torch.save(metrics, metrics_path)
        print(f"Saved metrics to {metrics_path}")
