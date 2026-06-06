#!/usr/bin/env python3
"""
make_FIHI_TS_animation.py

Create stitched FIHI_TS animations from 2 to 4 experiment directories.

This script:
1. finds common daily PNGs across the requested experiments,
2. filters them to selected months (default: AMJ = 04,05,06),
3. overlays a clean title band on each panel using Pillow,
4. stitches the panels together,
5. writes intermediate JPEG frames, and
6. calls ffmpeg to create an MP4 animation.

Directory assumptions
---------------------
Input PNG directories are assumed to be:

    ~/graphical/LD-pub-workspace/[sim_name]/FIHI_TS/[REGION]

with filenames like:

    YYYYMMDD_[sim_name]_FIHI_TS_binary-days.png

Outputs
-------
Frames:
    ~/graphical/LD-pub-workspace/frames/<RUN_TAG>/

Animation:
    ~/graphical/LD-pub-workspace/animations/<RUN_TAG>.mp4

Examples
--------
Default two simulations:
    python make_FIHI_TS_animation.py --region Aus

Explicit simulations:
    python make_FIHI_TS_animation.py --region Aus \
        LD-static-Cs1e-3 LD-static-Cs5e-4

Three-frame test:
    python make_FIHI_TS_animation.py --region Aus --nframes 3 \
        LD-static-Cs1e-3 LD-static-Cs5e-4

Different FPS:
    python make_FIHI_TS_animation.py --region Aus --fps 8 \
        LD-static-Cs1e-3 LD-static-Cs5e-4
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont


VALID_REGIONS = {"DML", "WIO", "EIO", "Aus", "VOL", "AS", "BS", "WS", "total"}
DEFAULT_SIMS = ["LD-static-Cs1e-3", "LD-static-Cs5e-4"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create stitched FIHI_TS MP4 animations from 2 to 4 experiments."
    )
    parser.add_argument(
        "sim_names",
        nargs="*",
        help="Simulation names (2 to 4). If omitted, defaults are used.",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Antarctic region, e.g. Aus",
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path.home() / "graphical" / "LD-pub-workspace"),
        help="Base working directory containing simulation subdirectories.",
    )
    parser.add_argument(
        "--field",
        default="FIHI_TS",
        help="Field name directory. Default: FIHI_TS",
    )
    parser.add_argument(
        "--method-tag",
        default="binary-days",
        help="Filename suffix tag. Default: binary-days",
    )
    parser.add_argument(
        "--months",
        default="04,05,06",
        help="Comma-separated list of months to include, e.g. 04,05,06",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Output animation frames per second. Default: 6",
    )
    parser.add_argument(
        "--title-band-px",
        type=int,
        default=70,
        help="Height of title band in pixels. Default: 70",
    )
    parser.add_argument(
        "--pointsize",
        type=int,
        default=42,
        help="Title font size. Default: 42",
    )
    parser.add_argument(
        "--title-bg",
        default="#eeeeee",
        help="Title band background colour. Default: #eeeeee",
    )
    parser.add_argument(
        "--title-fill",
        default="black",
        help="Title text colour. Default: black",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="Intermediate frame JPEG quality. Default: 92",
    )
    parser.add_argument(
        "--nframes",
        type=int,
        default=None,
        help="Optional limit on number of frames for testing/debug.",
    )
    parser.add_argument(
        "--fontfile",
        default=None,
        help="Optional path to a TrueType font file.",
    )
    return parser.parse_args()


def sanitise_name(text: str) -> str:
    """Convert a string into a filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def ymd_to_human(ymd: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"


def load_font(fontfile: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load a font for title rendering.

    Parameters
    ----------
    fontfile : str or None
        Path to a font file. If None, common DejaVu font locations are tried.
    size : int
        Requested font size in points.

    Returns
    -------
    PIL.ImageFont object
        Loaded font object. Falls back to Pillow default font if no TTF font is found.
    """
    candidates: List[Path] = []

    if fontfile:
        candidates.append(Path(fontfile))

    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)

    return ImageFont.load_default()


def validate_inputs(region: str, sim_names: Sequence[str], base_dir: Path, field: str) -> None:
    """
    Validate region and input directories.

    Parameters
    ----------
    region : str
        Antarctic region code.
    sim_names : sequence of str
        Simulation names.
    base_dir : pathlib.Path
        Root directory containing simulation subdirectories.
    field : str
        Field subdirectory name.

    Raises
    ------
    ValueError
        If region or simulation count is invalid.
    FileNotFoundError
        If a required input directory is missing.
    """
    if region not in VALID_REGIONS:
        raise ValueError(f"Invalid region '{region}'. Valid options: {sorted(VALID_REGIONS)}")

    if len(sim_names) < 2 or len(sim_names) > 4:
        raise ValueError("You must provide between 2 and 4 simulations.")

    for sim_name in sim_names:
        sim_dir = base_dir / sim_name / field / region
        print(sim_dir)
        if not sim_dir.is_dir():
            raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")


def find_common_dates(
    sim_names: Sequence[str],
    base_dir: Path,
    field: str,
    region: str,
    method_tag: str,
    months: set[str],
) -> List[str]:
    """
    Find dates common to all simulations for the requested months.

    Parameters
    ----------
    sim_names : sequence of str
        Simulation names.
    base_dir : pathlib.Path
        Root directory.
    field : str
        Field subdirectory.
    region : str
        Antarctic region code.
    method_tag : str
        Filename method tag, e.g. 'binary-days'.
    months : set of str
        Two-digit month strings to include, e.g. {'04', '05', '06'}.

    Returns
    -------
    list of str
        Sorted list of common YYYYMMDD strings.
    """
    first_sim = sim_names[0]
    first_dir = base_dir / first_sim / field / region
    pattern = f"20??????_{first_sim}_{field}_{method_tag}.png"

    common_dates: List[str] = []

    for file_path in sorted(first_dir.glob(pattern)):
        ymd = file_path.name[:8]
        mm = ymd[4:6]

        if not re.match(r"^20[0-9]{6}$", ymd):
            continue
        if mm not in months:
            continue

        exists_for_all = True
        for sim_name in sim_names[1:]:
            other = base_dir / sim_name / field / region / f"{ymd}_{sim_name}_{field}_{method_tag}.png"
            if not other.is_file():
                exists_for_all = False
                break

        if exists_for_all:
            common_dates.append(ymd)

    return common_dates


def draw_title_band(
    image: Image.Image,
    title_text: str,
    font: ImageFont.ImageFont,
    title_band_px: int,
    title_bg: str,
    title_fill: str,
) -> Image.Image:
    """
    Draw a title band on an image.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.
    title_text : str
        Text to draw in the title band.
    font : PIL.ImageFont
        Font object.
    title_band_px : int
        Height of the title band in pixels.
    title_bg : str
        Background colour.
    title_fill : str
        Text colour.

    Returns
    -------
    PIL.Image.Image
        Annotated RGB image.
    """
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, _ = image.size

    draw.rectangle((0, 0, width, title_band_px), fill=title_bg)

    bbox = draw.textbbox((0, 0), title_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = max(0, (width - text_width) // 2)
    y = max(0, (title_band_px - text_height) // 2 - 2)

    draw.text((x, y), title_text, fill=title_fill, font=font)
    return image


def pad_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """
    Pad an image to a target size with white background.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.
    width : int
        Target width.
    height : int
        Target height.

    Returns
    -------
    PIL.Image.Image
        Padded image.
    """
    if image.size == (width, height):
        return image
    out = Image.new("RGB", (width, height), "white")
    out.paste(image, (0, 0))
    return out


def hstack(images: Sequence[Image.Image]) -> Image.Image:
    """
    Horizontally concatenate images.

    Parameters
    ----------
    images : sequence of PIL.Image.Image
        Input images.

    Returns
    -------
    PIL.Image.Image
        Stacked image.
    """
    max_height = max(im.height for im in images)
    padded = [pad_image(im, im.width, max_height) for im in images]

    out = Image.new("RGB", (sum(im.width for im in padded), max_height), "white")
    x = 0
    for image in padded:
        out.paste(image, (x, 0))
        x += image.width
    return out


def vstack(images: Sequence[Image.Image]) -> Image.Image:
    """
    Vertically concatenate images.

    Parameters
    ----------
    images : sequence of PIL.Image.Image
        Input images.

    Returns
    -------
    PIL.Image.Image
        Stacked image.
    """
    max_width = max(im.width for im in images)
    padded = [pad_image(im, max_width, im.height) for im in images]

    out = Image.new("RGB", (max_width, sum(im.height for im in padded)), "white")
    y = 0
    for image in padded:
        out.paste(image, (0, y))
        y += image.height
    return out


def build_frames(
    sim_names: Sequence[str],
    dates: Sequence[str],
    base_dir: Path,
    field: str,
    region: str,
    method_tag: str,
    frame_dir: Path,
    font: ImageFont.ImageFont,
    title_band_px: int,
    title_bg: str,
    title_fill: str,
    jpeg_quality: int,
) -> None:
    """
    Build stitched JPEG frames from source PNGs.

    Parameters
    ----------
    sim_names : sequence of str
        Simulation names.
    dates : sequence of str
        Common YYYYMMDD dates.
    base_dir : pathlib.Path
        Root input directory.
    field : str
        Field subdirectory.
    region : str
        Antarctic region.
    method_tag : str
        Filename method tag.
    frame_dir : pathlib.Path
        Output frame directory.
    font : PIL.ImageFont
        Font object for title drawing.
    title_band_px : int
        Title band height in pixels.
    title_bg : str
        Title band background colour.
    title_fill : str
        Title text colour.
    jpeg_quality : int
        JPEG quality for frame output.
    """
    for old_frame in frame_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    total = len(dates)

    for idx, ymd in enumerate(dates, start=1):
        panels: List[Image.Image] = []
        date_human = ymd_to_human(ymd)

        for sim_name in sim_names:
            input_png = base_dir / sim_name / field / region / f"{ymd}_{sim_name}_{field}_{method_tag}.png"
            with Image.open(input_png) as image:
                panel = draw_title_band(
                    image.copy(),
                    f"{sim_name}    {date_human}",
                    font=font,
                    title_band_px=title_band_px,
                    title_bg=title_bg,
                    title_fill=title_fill,
                )
            panels.append(panel)

        if len(panels) in (2, 3):
            stitched = hstack(panels)
        elif len(panels) == 4:
            top = hstack(panels[:2])
            bottom = hstack(panels[2:])
            stitched = vstack([top, bottom])
        else:
            raise ValueError("Unsupported number of panels.")

        output_frame = frame_dir / f"frame_{idx:06d}.jpg"
        stitched.save(output_frame, quality=jpeg_quality, optimize=False, progressive=False)

        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"  wrote frame {idx:06d}/{total:06d}: {output_frame}", flush=True)


def encode_mp4(frame_dir: Path, out_mp4: Path, fps: int) -> None:
    """
    Encode a JPEG frame sequence into MP4 using ffmpeg.

    Parameters
    ----------
    frame_dir : pathlib.Path
        Directory containing frame_%06d.jpg.
    out_mp4 : pathlib.Path
        Output MP4 path.
    fps : int
        Output frame rate.

    Raises
    ------
    RuntimeError
        If ffmpeg fails or the output file is not created.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_mp4),
    ]

    subprocess.run(cmd, check=True)

    if not out_mp4.exists() or out_mp4.stat().st_size == 0:
        raise RuntimeError(f"MP4 was not created: {out_mp4}")


def main() -> None:
    """Main program entry point."""
    args = parse_args()

    sim_names = args.sim_names if args.sim_names else DEFAULT_SIMS
    months = {m.strip() for m in args.months.split(",") if m.strip()}

    base_dir = Path(args.base_dir).expanduser()
    frame_root = base_dir / "frames"
    anim_root = base_dir / "animations"
    frame_root.mkdir(parents=True, exist_ok=True)
    anim_root.mkdir(parents=True, exist_ok=True)

    validate_inputs(args.region, sim_names, base_dir, args.field)

    common_dates = find_common_dates(
        sim_names=sim_names,
        base_dir=base_dir,
        field=args.field,
        region=args.region,
        method_tag=args.method_tag,
        months=months,
    )

    if not common_dates:
        raise RuntimeError("No common dates found across selected simulations.")

    if args.nframes is not None:
        common_dates = common_dates[: args.nframes]

    first_year = common_dates[0][:4]
    last_year = common_dates[-1][:4]
    year_tag = first_year if first_year == last_year else f"{first_year}-{last_year}"

    run_tag = f"{sanitise_name('_'.join(sim_names))}_{args.region}_{args.field}_{''.join(sorted(months))}_{year_tag}"
    frame_dir = frame_root / run_tag
    frame_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = anim_root / f"{run_tag}.mp4"

    print()
    print(f"Region         : {args.region}")
    print(f"Field          : {args.field}")
    print(f"Sims           : {' '.join(sim_names)}")
    print(f"Common dates   : {len(common_dates)}")
    print(f"First date      : {common_dates[0]}")
    print(f"Last date       : {common_dates[-1]}")
    print(f"Frame dir       : {frame_dir}")
    print(f"Output MP4      : {out_mp4}")
    print()

    font = load_font(args.fontfile, args.pointsize)

    build_frames(
        sim_names=sim_names,
        dates=common_dates,
        base_dir=base_dir,
        field=args.field,
        region=args.region,
        method_tag=args.method_tag,
        frame_dir=frame_dir,
        font=font,
        title_band_px=args.title_band_px,
        title_bg=args.title_bg,
        title_fill=args.title_fill,
        jpeg_quality=args.jpeg_quality,
    )

    print()
    print("Encoding MP4 with ffmpeg...")
    print()

    encode_mp4(frame_dir=frame_dir, out_mp4=out_mp4, fps=args.fps)

    print()
    print("Done.")
    print("Animation written to:")
    print(f"  {out_mp4}")
    print()
    print("Frames written to:")
    print(f"  {frame_dir}")


if __name__ == "__main__":
    main()
