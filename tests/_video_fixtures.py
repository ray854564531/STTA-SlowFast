"""Synthetic mp4 generator for Kinetics pipeline tests."""
from pathlib import Path

import numpy as np


def make_mp4(path, num_frames=60, height=120, width=160, fps=30):
    """Write a synthetic mp4 with colored-noise frames.

    Returns the absolute path.
    """
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264',
                                 quality=8, macro_block_size=1)
    rng = np.random.default_rng(42)
    for _ in range(num_frames):
        frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        writer.append_data(frame)
    writer.close()
    return str(path)


def make_kinetics_tree(root, classes, videos_per_class=2, split='train',
                       num_frames=60):
    """Create root/<split>/<class>/<i>.mp4 layout."""
    for cls in classes:
        cls_dir = Path(root) / split / cls
        for i in range(videos_per_class):
            make_mp4(cls_dir / f'{i}.mp4', num_frames=num_frames)
    return str(Path(root) / split)
