"""
Pose Library Generator + Caller + Browser (standalone, self-contained)
========================================================================
Run this whole file once in Blender's Text Editor (Alt+P). It will:
  1. Generate a FIXED (reproducible -- same seed every run) set of pose
     variants per armature currently in the scene.
  2. Add a sidebar panel in the 3D Viewport (press N, look for the
     "Pose Library" tab) to browse those poses, and to actually CALL
     (duplicate + pose + place) a character at a given location.

Responsibility split (settled 2026-07): this file owns "given a pose-
library entry and a target location, make that character exist there,
posed correctly." The arrangement/combination logic (which entry goes at
which location for a given people-count x distance condition) is a
SEPARATE piece, not built yet -- it will call call_pose() from here
rather than re-implementing armature/bone handling itself.

This file does not read or depend on any previously-written script --
everything it needs is defined here.
"""

import bpy
import math
import random
import json
import os

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
POSES_PER_CHARACTER = 20
RANDOM_SEED = 42          # fixed on purpose -- same seed = same pose set every run

# Jitter magnitude, PER AXIS. Axis order matches rotation_euler: (X,Y,Z).
# Tuned interactively against the actual 5-person / 70cm layout in
# Blender (2026-07) -- these are no longer guesses:
#   X: excluded-positive-direction bones confirmed via axis diagnostic
#      (12-pose viewport check) to swing the arm backward -- excluded
#      entirely, not just reduced. Magnitude narrowed twice (8 -> 5 ->
#      2.5 deg) after visually checking arms didn't reach into the
#      neighboring character at 70cm spacing.
#   Y: same diagnostic; positive excluded on LeftArm/LeftForeArm only
#      (RightArm's positive Y was checked and NOT flagged).
#   Z: one-sided (0 to max, same sign as that bone's own base value) --
#      original reasoning was "avoid swinging into the torso"; diagnostic
#      didn't actually flag Z as the culprit (X was), but one-sided
#      Z was never shown to be wrong either, so it's kept as-is.
JITTER_X_DEG = 2.5
JITTER_Y_DEG = 2.0
JITTER_Z_MAX_DEG = 6.0
EXCLUDE_POSITIVE_X = {"LeftArm", "RightArm"}
EXCLUDE_POSITIVE_Y = {"LeftArm", "LeftForeArm"}

# Only the 4 major arm bones are posed. LeftShoulder/RightShoulder
# (collar bones) came back at ~0 rotation in the neutral-stance
# calibration -- jittering a ~0 baseline barely shows, left out to keep
# this simpler.
POSE_BONES = ["LeftArm", "LeftForeArm", "RightArm", "RightForeArm"]

# Neutral relaxed-arms starting point (median of 5 real Mixamo characters,
# hand-posed and read back in Blender, 2026-07). Jitter is layered on top
# of this per generated pose.
NEUTRAL_BASE = {
    "LeftArm":      (1.4138, 0.0228, 0.1523),
    "LeftForeArm":  (0.088, 0.0064, -0.124),
    "RightArm":     (1.3848, -0.0432, -0.2013),
    "RightForeArm": (0.0887, -0.0057, 0.1129),
}

# Facing/rotation: deliberately NOT handled here -- see call_pose()
# docstring. Every call gets rotation (0,0,0) until a real render
# confirms what's actually needed.

TEMPLATE_COLLECTION_NAME = "PoseTemplates"
GENERATED_COLLECTION_NAME = "Generated"


# ---------------------------------------------------------------------
# Collections -- separates the original characters (kept only as
# never-rendered templates to duplicate FROM) from everything call_pose()
# creates, so cleanup can never accidentally touch templates, camera,
# lights, or anything else in the scene.
# ---------------------------------------------------------------------
def ensure_collections():
    templates = bpy.data.collections.get(TEMPLATE_COLLECTION_NAME)
    if templates is None:
        templates = bpy.data.collections.new(TEMPLATE_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(templates)
    templates.hide_render = True

    generated = bpy.data.collections.get(GENERATED_COLLECTION_NAME)
    if generated is None:
        generated = bpy.data.collections.new(GENERATED_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(generated)
    return templates, generated


def setup_templates():
    """One-time setup: finds every armature currently sitting loose in
    the scene (and its mesh children), moves the whole hierarchy into
    PoseTemplates (hidden from render). Safe to call more than once --
    objects already in the collection are skipped."""
    templates, _ = ensure_collections()
    moved = []
    for obj in list(bpy.data.objects):
        if obj.type != 'ARMATURE':
            continue
        if obj.name in templates.objects:
            continue
        group = [obj] + [c for c in obj.children if c.type == 'MESH']
        for o in group:
            for coll in list(o.users_collection):
                coll.objects.unlink(o)
            templates.objects.link(o)
        moved.append(obj.name)
    if moved:
        print(f"[templates] moved into '{TEMPLATE_COLLECTION_NAME}' (hidden from render): {moved}")
    else:
        print("[templates] nothing new to move -- already set up")


def clear_generated():
    """Debug-only manual cleanup: deletes everything in the Generated
    collection and nothing else. Safe by construction -- call_pose() is
    the only thing that ever puts objects there."""
    _, generated = ensure_collections()
    objs = list(generated.objects)
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)
    print(f"[generated] cleared {len(objs)} objects")


# ---------------------------------------------------------------------
# Mixamo bone-namespace detection -- Blender renames "mixamorig:" to
# "mixamorig2:", "mixamorig8:" etc. as soon as more than one Mixamo
# armature exists in the same scene (confirmed empirically, 2026-07).
# Never assume the prefix; detect it per armature.
# ---------------------------------------------------------------------
def detect_mixamo_prefix(armature_obj):
    for bone in armature_obj.pose.bones:
        if bone.name.startswith("mixamorig") and ":" in bone.name:
            return bone.name.split(":")[0] + ":"
    return None


# ---------------------------------------------------------------------
# Pose generation -- flat list, one entry per (armature, pose_index)
# ---------------------------------------------------------------------
POSE_LIBRARY = []
_LAST_GENERATED_MODE = None  # "library" or "diagnostic" -- export checks this


def _jittered_angle(bone_name, base_triplet, rng):
    bx, by, bz = base_triplet
    x_jit = math.radians(JITTER_X_DEG)
    y_jit = math.radians(JITTER_Y_DEG)

    if bone_name in EXCLUDE_POSITIVE_X:
        new_x = bx - rng.uniform(0.0, x_jit)
    else:
        new_x = bx + rng.uniform(-x_jit, x_jit)

    if bone_name in EXCLUDE_POSITIVE_Y:
        new_y = by - rng.uniform(0.0, y_jit)
    else:
        new_y = by + rng.uniform(-y_jit, y_jit)

    sign = 1.0 if bz >= 0 else -1.0
    new_z = bz + sign * rng.uniform(0.0, math.radians(JITTER_Z_MAX_DEG))
    return (new_x, new_y, new_z)


def generate_pose_library():
    POSE_LIBRARY.clear()
    global _LAST_GENERATED_MODE
    _LAST_GENERATED_MODE = "library"
    rng = random.Random(RANDOM_SEED)
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    for arm in armatures:
        prefix = detect_mixamo_prefix(arm)
        if prefix is None:
            print(f"[skip] {arm.name}: no mixamorig-style bones found")
            continue
        for pose_idx in range(POSES_PER_CHARACTER):
            angles = {
                bone_name: _jittered_angle(bone_name, NEUTRAL_BASE[bone_name], rng)
                for bone_name in POSE_BONES
            }
            POSE_LIBRARY.append({
                "id": f"{arm.name}_pose{pose_idx:02d}",
                "armature_name": arm.name,
                "prefix": prefix,
                "pose_index": pose_idx,
                "angles": angles,
            })
    print(f"[pose library] generated {len(POSE_LIBRARY)} poses across {len(armatures)} armatures")


def apply_pose(entry):
    """Preview only -- poses the TEMPLATE armature directly (not a
    duplicate). Used for fast Previous/Next browsing. Harmless even
    though templates are hidden from render: whatever pose a template
    happens to be showing has no effect on call_pose(), which always
    sets bone angles explicitly on the new duplicate right after
    creating it."""
    arm = bpy.data.objects.get(entry["armature_name"])
    if arm is None:
        return
    prefix = entry["prefix"]
    for bone_name, euler in entry["angles"].items():
        pbone = arm.pose.bones.get(prefix + bone_name)
        if pbone is None:
            continue
        pbone.rotation_mode = 'XYZ'
        pbone.rotation_euler = euler
    bpy.context.view_layer.update()


# ---------------------------------------------------------------------
# THE CALL FUNCTION -- given a pose-library entry and a target location,
# duplicates the matching template (armature + all mesh children,
# together, via bpy.ops.object.duplicate() so Blender correctly re-links
# the Armature modifier to the NEW copy instead of leaving it pointed at
# the template), poses it, places it, faces it -Y, and files it under
# the Generated collection.
# ---------------------------------------------------------------------
def call_pose(entry, location):
    """Returns the newly created armature object. Its mesh children come
    along with it.

    Rotation is always (0, 0, 0) -- no facing bias applied. An earlier
    version hardcoded a 180 deg Z rotation based on eyeballing the 3D
    Viewport's axis gizmo, but that's not a trustworthy way to determine
    "does the character face the camera": the camera itself is tilted
    (X rotation ~82 deg), so world -Y in the viewport doesn't obviously
    correspond to "toward the camera lens." Decided (2026-07) to drop
    the assumption entirely rather than keep a guess that might be
    backwards. The correct way to check this is a real render (F12) once
    the arrangement generator can actually place someone and show
    whether you see their face or the back of their head -- not reading
    Euler numbers in the viewport. Add rotation handling back here once
    that's been confirmed with an actual render, not before.

    Uses the data API directly (obj.copy()) rather than
    bpy.ops.object.duplicate() -- the operator version was confirmed
    (2026-07) to sometimes end up moving/only-partially-duplicating the
    TEMPLATE itself (mesh children got duplicated, the armature did not,
    most likely because context.view_layer.objects.active after the
    operator call didn't reliably point at the new copy). This version
    has no dependency on bpy.context selection/active-object state at
    all, so there's no ambiguity about what gets duplicated or moved.
    """
    templates, generated = ensure_collections()
    template_arm = bpy.data.objects.get(entry["armature_name"])
    if template_arm is None or template_arm.name not in templates.objects:
        raise KeyError(
            f"Template '{entry['armature_name']}' not found in the "
            f"'{TEMPLATE_COLLECTION_NAME}' collection -- run Setup "
            f"Templates first."
        )

    # 1) Duplicate the armature OBJECT. Armature DATA (the bone rest
    #    structure) stays shared/linked with the template -- that's fine
    #    and saves memory, since actual pose state (what we're about to
    #    set) lives per-OBJECT in Blender, not on the shared data-block.
    new_arm = template_arm.copy()
    new_arm.animation_data_clear()  # guards against a leftover Action
                                     # overriding our manual pose at
                                     # render time -- same class of bug
                                     # as the earlier T-pose-on-render issue
    generated.objects.link(new_arm)

    # 2) Duplicate each mesh child AT THE TEMPLATE'S CURRENT (unmoved)
    #    position, re-pointing both the parent relationship and the
    #    Armature modifier to the NEW armature. bpy.ops.duplicate() would
    #    normally handle this automatically; done explicitly here since
    #    it's bypassed entirely now.
    for mesh_child in template_arm.children:
        if mesh_child.type != 'MESH':
            continue
        new_mesh_obj = mesh_child.copy()  # mesh DATA also stays shared --
                                           # fine, geometry is never edited
        new_mesh_obj.parent = new_arm
        new_mesh_obj.matrix_parent_inverse = new_arm.matrix_world.inverted()
        for mod in new_mesh_obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object == template_arm:
                mod.object = new_arm
        generated.objects.link(new_mesh_obj)

    # 3) ONLY NOW move/pose the duplicate -- the template's own transform
    #    and pose are never touched anywhere in this function.
    new_arm.location = location
    new_arm.rotation_euler = (0.0, 0.0, 0.0)

    prefix = entry["prefix"]
    for bone_name, euler in entry["angles"].items():
        pbone = new_arm.pose.bones.get(prefix + bone_name)
        if pbone is None:
            continue
        pbone.rotation_mode = 'XYZ'
        pbone.rotation_euler = euler

    bpy.context.view_layer.update()
    return new_arm


# ---------------------------------------------------------------------
# Axis diagnostic -- isolates ONE rotation axis at a time (all others
# held at the neutral base) on the first armature found, so you can look
# at the viewport and see directly what X/Y/Z each actually do.
# ---------------------------------------------------------------------
DIAG_BONES = ["LeftArm", "RightArm"]
DIAG_AXES = ["X", "Y", "Z"]
DIAG_ANGLE_DEG = 30.0
_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def generate_axis_diagnostic():
    POSE_LIBRARY.clear()
    global _LAST_GENERATED_MODE
    _LAST_GENERATED_MODE = "diagnostic"
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if not armatures:
        print("[diagnostic] no armature found")
        return
    arm = armatures[0]
    prefix = detect_mixamo_prefix(arm)
    if prefix is None:
        print(f"[diagnostic] {arm.name}: no mixamorig-style bones found")
        return
    for bone_name in DIAG_BONES:
        for axis in DIAG_AXES:
            for sign, sign_label in [(1, "+"), (-1, "-")]:
                angles = {b: NEUTRAL_BASE[b] for b in POSE_BONES}
                new_val = list(NEUTRAL_BASE[bone_name])
                new_val[_AXIS_INDEX[axis]] += sign * math.radians(DIAG_ANGLE_DEG)
                angles[bone_name] = tuple(new_val)
                POSE_LIBRARY.append({
                    "id": f"DIAG_{bone_name}_{axis}{sign_label}{int(DIAG_ANGLE_DEG)}",
                    "armature_name": arm.name,
                    "prefix": prefix,
                    "pose_index": len(POSE_LIBRARY),
                    "angles": angles,
                })
    print(f"[diagnostic] {len(POSE_LIBRARY)} single-axis test poses on {arm.name}.")


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
def export_pose_library(filepath):
    data = {
        "random_seed": RANDOM_SEED,
        "poses_per_character": POSES_PER_CHARACTER,
        "jitter_x_deg": JITTER_X_DEG,
        "jitter_y_deg": JITTER_Y_DEG,
        "jitter_z_max_deg": JITTER_Z_MAX_DEG,
        "exclude_positive_x": sorted(EXCLUDE_POSITIVE_X),
        "exclude_positive_y": sorted(EXCLUDE_POSITIVE_Y),
        "entries": [
            {
                "id": e["id"],
                "armature_name": e["armature_name"],
                "pose_index": e["pose_index"],
                "angles": {bone: list(euler) for bone, euler in e["angles"].items()},
            }
            for e in POSE_LIBRARY
        ],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[pose library] exported {len(POSE_LIBRARY)} entries to {filepath}")


# ---------------------------------------------------------------------
# UI state
# ---------------------------------------------------------------------
def _on_index_change(self, context):
    if not POSE_LIBRARY:
        return
    n = len(POSE_LIBRARY)
    if self.poselib_index >= n:
        self.poselib_index = n - 1
        return
    if self.poselib_index < 0:
        self.poselib_index = 0
        return
    apply_pose(POSE_LIBRARY[self.poselib_index])


# ---------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------
class POSELIB_OT_setup_templates(bpy.types.Operator):
    bl_idname = "poselib.setup_templates"
    bl_label = "Setup Templates"
    bl_description = "Move existing armatures into PoseTemplates (hidden from render)"

    def execute(self, context):
        setup_templates()
        return {'FINISHED'}


class POSELIB_OT_generate(bpy.types.Operator):
    bl_idname = "poselib.generate"
    bl_label = "Generate Pose Library"
    bl_description = f"Generate {POSES_PER_CHARACTER} poses per armature (fixed seed, reproducible)"

    def execute(self, context):
        generate_pose_library()
        if POSE_LIBRARY:
            context.scene.poselib_index = 0
            apply_pose(POSE_LIBRARY[0])
        else:
            self.report({'WARNING'}, "No Mixamo armatures found in the scene")
        return {'FINISHED'}


class POSELIB_OT_diagnostic(bpy.types.Operator):
    bl_idname = "poselib.diagnostic"
    bl_label = "Generate Axis Diagnostic"
    bl_description = "Isolate each rotation axis one at a time to see what it actually does"

    def execute(self, context):
        generate_axis_diagnostic()
        if POSE_LIBRARY:
            context.scene.poselib_index = 0
            apply_pose(POSE_LIBRARY[0])
        else:
            self.report({'WARNING'}, "No Mixamo armatures found in the scene")
        return {'FINISHED'}


class POSELIB_OT_export(bpy.types.Operator):
    bl_idname = "poselib.export"
    bl_label = "Export Pose Library (JSON)"
    bl_description = "Save the generated pose library to a JSON file next to this .blend file"

    def execute(self, context):
        if not POSE_LIBRARY:
            self.report({'WARNING'}, "Nothing to export -- generate the library first")
            return {'CANCELLED'}
        if _LAST_GENERATED_MODE != "library":
            self.report({'WARNING'}, "Current data is from Axis Diagnostic, not the pose "
                                      "library -- run 'Generate Pose Library' first")
            return {'CANCELLED'}
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else bpy.app.tempdir
        filepath = os.path.join(blend_dir, "pose_library.json")
        export_pose_library(filepath)
        self.report({'INFO'}, f"Exported to {filepath}")
        return {'FINISHED'}


class POSELIB_OT_test_call(bpy.types.Operator):
    bl_idname = "poselib.test_call"
    bl_label = "Test Call (duplicate + place)"
    bl_description = "Duplicate the currently browsed pose in-place (no offset, no rotation), to verify call_pose() works"

    def execute(self, context):
        if not POSE_LIBRARY:
            self.report({'WARNING'}, "Generate the library first")
            return {'CANCELLED'}
        entry = POSE_LIBRARY[context.scene.poselib_index]
        try:
            new_arm = call_pose(entry, location=(0.0, 0.0, 0.0))
        except KeyError as e:
            self.report({'WARNING'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Called {entry['id']} -> {new_arm.name}")
        return {'FINISHED'}


class POSELIB_OT_clear_generated(bpy.types.Operator):
    bl_idname = "poselib.clear_generated"
    bl_label = "Clear Generated (debug)"
    bl_description = "Delete everything in the Generated collection. Never touches templates."

    def execute(self, context):
        clear_generated()
        return {'FINISHED'}


class POSELIB_OT_next(bpy.types.Operator):
    bl_idname = "poselib.next"
    bl_label = "Next"

    def execute(self, context):
        if not POSE_LIBRARY:
            self.report({'WARNING'}, "Generate the library first")
            return {'CANCELLED'}
        context.scene.poselib_index += 1
        return {'FINISHED'}


class POSELIB_OT_prev(bpy.types.Operator):
    bl_idname = "poselib.prev"
    bl_label = "Previous"

    def execute(self, context):
        if not POSE_LIBRARY:
            self.report({'WARNING'}, "Generate the library first")
            return {'CANCELLED'}
        context.scene.poselib_index -= 1
        return {'FINISHED'}


# ---------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------
class POSELIB_PT_panel(bpy.types.Panel):
    bl_label = "Pose Library"
    bl_idname = "POSELIB_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Pose Library"

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Setup (run once)")
        box.operator("poselib.setup_templates", icon='OUTLINER_COLLECTION')

        layout.separator()
        layout.operator("poselib.generate", icon='FILE_REFRESH')
        layout.operator("poselib.diagnostic", icon='TOOL_SETTINGS')
        layout.separator()

        if not POSE_LIBRARY:
            layout.label(text="Not generated yet.")
            return

        entry = POSE_LIBRARY[context.scene.poselib_index]
        layout.label(text=f"{context.scene.poselib_index + 1} / {len(POSE_LIBRARY)}")
        layout.label(text=entry["id"])
        layout.prop(context.scene, "poselib_index", text="Go to #")

        row = layout.row(align=True)
        row.operator("poselib.prev", text="< Previous")
        row.operator("poselib.next", text="Next >")

        layout.separator()
        layout.operator("poselib.export", icon='EXPORT')

        layout.separator()
        box2 = layout.box()
        box2.label(text="Call pipeline (debug)")
        box2.operator("poselib.test_call", icon='DUPLICATE')
        box2.operator("poselib.clear_generated", icon='TRASH')


CLASSES = [
    POSELIB_OT_setup_templates,
    POSELIB_OT_generate,
    POSELIB_OT_diagnostic,
    POSELIB_OT_export,
    POSELIB_OT_test_call,
    POSELIB_OT_clear_generated,
    POSELIB_OT_next,
    POSELIB_OT_prev,
    POSELIB_PT_panel,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.poselib_index = bpy.props.IntProperty(default=0, min=0, update=_on_index_change)


def unregister():
    for cls in CLASSES:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.poselib_index


# Auto-run when this file is executed via Alt+P in the Text Editor.
try:
    unregister()
except Exception:
    pass
register()
generate_pose_library()
if POSE_LIBRARY:
    bpy.context.scene.poselib_index = 0
    apply_pose(POSE_LIBRARY[0])

print("Pose Library ready -- press N in the 3D Viewport, open the 'Pose Library' tab. "
      "Click 'Setup Templates' first (one-time) before using Test Call.")