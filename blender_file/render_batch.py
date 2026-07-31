"""
render_batch.py -- item 4: reads arrangement.json, renders every sample
==========================================================================
Run this in the SAME Blender session where pose_library_generator.py has
already been run with "Generate Pose Library" (not Axis Diagnostic).
This script calls its functions directly -- call_pose(), clear_generated(),
POSE_LIBRARY, ensure_collections() -- none of that is redefined here.

PREREQUISITE: the pose_entry_id values in arrangement.json only make
sense if POSE_LIBRARY in this session was generated with the same
RANDOM_SEED (42, unchanged in pose_library_generator.py) that was active
when arrangement.json was built. If you closed/reopened Blender, re-run
"Generate Pose Library" before running this script.

Per sample in arrangement.json, this script:
  1. Sets the camera to that sample's distance (Y axis, confirmed
     2026-07: distance_cm=250 -> Y=-2.5m, i.e. Y_m = -distance_cm/100).
  2. call_pose()'s each person in the sample at (x_m, 0, 0).
  3. Renders to PNG.
  4. clear_generated() -- wipes this sample's people before the next one.

Images only -- no ground truth JSON. Decided 2026-07: the analysis
methodology switched to FP32-vs-INT8/FP16 comparison (the standard
approach in the quantization literature -- same convention the original
real-footage study already used), computed downstream once these images
are actually run through the Nano's TensorRT engines (separate scope,
handled elsewhere). Geometric ground truth is no longer the comparison
basis, so this script doesn't compute or write it.

Does NOT touch scene lighting/world/render engine -- your scene already
renders correctly (confirmed via your own F12 tests), so this only
touches the camera and the render trigger itself.
"""

import bpy
import json
import math
import os

# ---------------------------------------------------------------------
# CONFIG -- check these before running
# ---------------------------------------------------------------------
CAMERA_NAME = "攝影機"  # matches this project's actual camera object name
ARRANGEMENT_JSON_PATH = bpy.path.abspath("//arrangement.json")
OUTPUT_DIR = bpy.path.abspath("//render_output")

CAMERA_FOV_X_DEG = 70.5     # confirmed C920-based horizontal FOV
CAMERA_Z_M = 1.2            # confirmed camera height
CAMERA_ROTATION_X_DEG = 82.4  # confirmed camera pitch
RESOLUTION = (1280, 720)


def distance_cm_to_camera_y(distance_cm):
    return -distance_cm / 100.0


def _load_pose_library_module():
    """Ensures call_pose(), clear_generated(), POSE_LIBRARY, etc. exist
    in THIS script's namespace.

    Confirmed 2026-07: globals from a separately-run Text Editor script
    are NOT reliably available to a different script run afterward, even
    in the same Blender session. This loads that file's actual code
    explicitly instead, so there's no dependency on what happened to run
    earlier.

    Tries the currently-open Text Editor tab first (if
    pose_library_generator.py is open there), falls back to reading the
    file from disk next to this .blend file.
    """
    text_block = bpy.data.texts.get("pose_library_generator.py")
    if text_block is not None:
        source = text_block.as_string()
        print("[render_batch] loaded pose_library_generator.py from the open Text Editor tab")
    else:
        path = bpy.path.abspath("//pose_library_generator.py")
        if not os.path.exists(path):
            raise RuntimeError(
                "Can't find pose_library_generator.py -- either open it as "
                "a tab in the Text Editor (any name is fine as long as "
                "it's open), or save the file next to this .blend file."
            )
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        print(f"[render_batch] loaded pose_library_generator.py from {path}")
    exec(source, globals())


_load_pose_library_module()


# ---------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------
def load_arrangement(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pose_lookup():
    return {e["id"]: e for e in POSE_LIBRARY}  # noqa: F821 -- from pose_library_generator.py


def setup_camera_and_render_settings():
    cam_obj = bpy.data.objects.get(CAMERA_NAME)
    if cam_obj is None:
        raise RuntimeError(
            f"Camera object '{CAMERA_NAME}' not found in this scene -- "
            f"edit CAMERA_NAME at the top of this script to match your "
            f"actual camera's name in the Outliner."
        )
    cam_obj.data.sensor_fit = 'HORIZONTAL'
    cam_obj.data.lens_unit = 'FOV'
    cam_obj.data.angle = math.radians(CAMERA_FOV_X_DEG)
    cam_obj.rotation_euler = (math.radians(CAMERA_ROTATION_X_DEG), 0.0, 0.0)

    scene = bpy.context.scene
    scene.camera = cam_obj
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    return cam_obj, scene


def render_sample(sample, cam_obj, scene, pose_lookup):
    cam_obj.location = (0.0, distance_cm_to_camera_y(sample["distance_cm"]), CAMERA_Z_M)

    for person in sample["people"]:
        entry = pose_lookup.get(person["pose_entry_id"])
        if entry is None:
            print(f"  [WARN] pose entry '{person['pose_entry_id']}' not in POSE_LIBRARY "
                  f"-- re-run 'Generate Pose Library' with the same seed first")
            continue
        call_pose(entry, location=(person["x_m"], 0.0, 0.0))  # noqa: F821

    img_path = os.path.join(OUTPUT_DIR, "images", f"{sample['sample_id']}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)

    clear_generated()  # noqa: F821 -- from pose_library_generator.py


def run_batch(limit=None):
    arrangement = load_arrangement(ARRANGEMENT_JSON_PATH)
    pose_lookup = build_pose_lookup()
    if not pose_lookup:
        raise RuntimeError(
            "POSE_LIBRARY is empty -- run 'Generate Pose Library' in "
            "pose_library_generator.py first, in THIS Blender session."
        )

    cam_obj, scene = setup_camera_and_render_settings()

    samples = arrangement if limit is None else arrangement[:limit]
    print(f"Rendering {len(samples)} of {len(arrangement)} total samples")

    n_failed = 0
    for i, sample in enumerate(samples):
        print(f"[{i + 1}/{len(samples)}] {sample['sample_id']}")
        try:
            render_sample(sample, cam_obj, scene, pose_lookup)
        except Exception as e:
            n_failed += 1
            print(f"  [FAILED] {sample['sample_id']}: {e}")

    print(f"done. {len(samples) - n_failed} succeeded, {n_failed} failed.")
    print(f"images: {os.path.join(OUTPUT_DIR, 'images')}")


# Runs the FULL batch (no pilot limit -- confirmed no longer needed).
# To render only a subset instead, call run_batch(limit=N) from the
# Python Console.
run_batch()