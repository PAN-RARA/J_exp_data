import bpy

bones = ["LeftArm","LeftForeArm","RightArm","RightForeArm","LeftShoulder","RightShoulder"]

report = []
armature_count = 0
for obj in bpy.data.objects:
    if obj.type != 'ARMATURE':
        continue
    armature_count += 1
    result = {}
    missing = []
    for b in bones:
        pb = obj.pose.bones.get("mixamorig:" + b)
        if pb is None:
            missing.append(b)
            continue
        pb.rotation_mode = 'XYZ'
        result[b] = tuple(round(v, 4) for v in pb.rotation_euler)
    report.append(f"--- {obj.name} ---\n{result}\nmissing: {missing}\n")

report.insert(0, f"[found {armature_count} armature objects]\n")

# 直接寫成一個新的文字資料塊,在Text Editor的下拉選單裡就能打開看
text_block = bpy.data.texts.new("pose_report")
text_block.write("\n".join(report))
print("done, check the 'pose_report' text block in the Text Editor dropdown")