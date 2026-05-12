import bpy
import os
import json
import sys
from pathlib import Path

# ---------------------------------------------------
# ROOT
# ---------------------------------------------------

ROOT = "/Users/hrithikg/SWRepos/switchlight"

# ---------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------

def get_arg(name, default):
    if "--" not in sys.argv:
        return default

    argv = sys.argv[sys.argv.index("--") + 1:]

    if name in argv:
        idx = argv.index(name)
        if idx + 1 < len(argv):
            return argv[idx + 1]

    return default


HDRI_REL_PATH = get_arg(
    "--hdri",
    "data/blender/hdri/studio_small_03_2k.hdr"
)

OUTPUT_REL_DIR = get_arg(
    "--outdir",
    "data/blender/smoke_test"
)

FRAME_ID = get_arg(
    "--frame",
    "0000"
)

HDRI_PATH = os.path.join(ROOT, HDRI_REL_PATH)
OUTPUT_DIR = os.path.join(ROOT, OUTPUT_REL_DIR)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("========== RENDER CONFIG ==========")
print("HDRI:", HDRI_PATH)
print("OUTPUT:", OUTPUT_DIR)
print("FRAME:", FRAME_ID)
print("===================================")

scene = bpy.context.scene

# ---------------------------------------------------
# BASIC RENDER SETTINGS
# ---------------------------------------------------

scene.render.engine = "CYCLES"
scene.cycles.samples = 64

scene.render.resolution_x = 768
scene.render.resolution_y = 768
scene.render.resolution_percentage = 100

scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1

scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

# ---------------------------------------------------
# HDRI WORLD
# ---------------------------------------------------

world = bpy.data.worlds.get("World")

if world is None:
    world = bpy.data.worlds.new("World")

scene.world = world
world.use_nodes = True

nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

env_tex = nodes.new(type="ShaderNodeTexEnvironment")
env_tex.image = bpy.data.images.load(HDRI_PATH)

bg = nodes.new(type="ShaderNodeBackground")
bg.inputs["Strength"].default_value = 1.0

out = nodes.new(type="ShaderNodeOutputWorld")

links.new(env_tex.outputs["Color"], bg.inputs["Color"])
links.new(bg.outputs["Background"], out.inputs["Surface"])

# ---------------------------------------------------
# PASSES
# ---------------------------------------------------

view_layer = bpy.context.view_layer
view_layer.use_pass_normal = True
view_layer.use_pass_diffuse_color = True

# ---------------------------------------------------
# CAMERA FALLBACK
# ---------------------------------------------------

if scene.camera is None:
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    scene.camera = cam

    cam.location = (0, -4, 1.6)
    cam.rotation_euler = (1.3, 0, 0)

# ---------------------------------------------------
# COMPOSITOR HELPERS
# ---------------------------------------------------

def clear_compositor():
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    return tree

# ---------------------------------------------------
# BEAUTY / WORLD NORMAL / MASK
# ---------------------------------------------------

tree = clear_compositor()

render_layers = tree.nodes.new("CompositorNodeRLayers")

beauty_out = tree.nodes.new("CompositorNodeOutputFile")
beauty_out.base_path = OUTPUT_DIR
beauty_out.file_slots[0].path = f"beauty_"

normal_out = tree.nodes.new("CompositorNodeOutputFile")
normal_out.base_path = OUTPUT_DIR
normal_out.file_slots[0].path = f"normal_world_"

mask_out = tree.nodes.new("CompositorNodeOutputFile")
mask_out.base_path = OUTPUT_DIR
mask_out.file_slots[0].path = f"mask_"

tree.links.new(render_layers.outputs["Image"], beauty_out.inputs[0])
tree.links.new(render_layers.outputs["Normal"], normal_out.inputs[0])
tree.links.new(render_layers.outputs["Alpha"], mask_out.inputs[0])

scene.render.filepath = os.path.join(
    OUTPUT_DIR,
    f"beauty_direct.png"
)

bpy.ops.render.render(write_still=True)

# ---------------------------------------------------
# STORE ORIGINAL MATERIALS
# ---------------------------------------------------

original_materials = {}

for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        original_materials[obj.name] = [
            slot.material for slot in obj.material_slots
        ]

# ---------------------------------------------------
# ALBEDO OVERRIDE
# ---------------------------------------------------

for obj in bpy.context.scene.objects:
    if obj.type != "MESH":
        continue

    for slot in obj.material_slots:
        mat = slot.material

        if mat is None:
            continue

        mat.use_nodes = True

        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links

        principled = None
        output = None

        for node in nodes:
            if node.type == "BSDF_PRINCIPLED":
                principled = node
            elif node.type == "OUTPUT_MATERIAL":
                output = node

        if principled is None or output is None:
            continue

        base_color_input = principled.inputs.get("Base Color")

        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0

        if base_color_input and base_color_input.is_linked:
            source_socket = base_color_input.links[0].from_socket
            links.new(source_socket, emission.inputs["Color"])

        elif base_color_input:
            emission.inputs["Color"].default_value = (
                base_color_input.default_value
            )

        for link in list(output.inputs["Surface"].links):
            links.remove(link)

        links.new(
            emission.outputs["Emission"],
            output.inputs["Surface"]
        )

# ---------------------------------------------------
# ALBEDO RENDER
# ---------------------------------------------------

tree = clear_compositor()

render_layers = tree.nodes.new("CompositorNodeRLayers")

albedo_out = tree.nodes.new("CompositorNodeOutputFile")
albedo_out.base_path = OUTPUT_DIR
albedo_out.file_slots[0].path = f"albedo_"

tree.links.new(render_layers.outputs["Image"], albedo_out.inputs[0])

scene.render.filepath = os.path.join(
    OUTPUT_DIR,
    f"albedo_direct.png"
)

bpy.ops.render.render(write_still=True)

# ---------------------------------------------------
# RESTORE MATERIALS
# ---------------------------------------------------

for obj in bpy.context.scene.objects:
    if obj.type != "MESH":
        continue

    if obj.name not in original_materials:
        continue

    obj.data.materials.clear()

    for mat in original_materials[obj.name]:
        obj.data.materials.append(mat)

# ---------------------------------------------------
# CAMERA-SPACE NORMALS
# ---------------------------------------------------

def setup_camera_space_normal_materials():

    for obj in bpy.context.scene.objects:

        if obj.type != "MESH":
            continue

        normal_mat = bpy.data.materials.new(
            name=f"{obj.name}_camera_normal"
        )

        normal_mat.use_nodes = True

        nt = normal_mat.node_tree
        nodes = nt.nodes
        links = nt.links

        nodes.clear()

        geom = nodes.new(type="ShaderNodeNewGeometry")

        vec_transform = nodes.new(type="ShaderNodeVectorTransform")
        vec_transform.vector_type = "VECTOR"
        vec_transform.convert_from = "WORLD"
        vec_transform.convert_to = "CAMERA"

        separate = nodes.new(type="ShaderNodeSeparateXYZ")

        combine = nodes.new(type="ShaderNodeCombineXYZ")

        emission = nodes.new(type="ShaderNodeEmission")

        output = nodes.new(type="ShaderNodeOutputMaterial")

        def remap_channel(channel_output):

            mul = nodes.new(type="ShaderNodeMath")
            mul.operation = "MULTIPLY"
            mul.inputs[1].default_value = 0.5

            add = nodes.new(type="ShaderNodeMath")
            add.operation = "ADD"
            add.inputs[1].default_value = 0.5

            links.new(channel_output, mul.inputs[0])
            links.new(mul.outputs[0], add.inputs[0])

            return add.outputs[0]

        links.new(
            geom.outputs["Normal"],
            vec_transform.inputs["Vector"]
        )

        links.new(
            vec_transform.outputs["Vector"],
            separate.inputs["Vector"]
        )

        links.new(
            remap_channel(separate.outputs["X"]),
            combine.inputs["X"]
        )

        links.new(
            remap_channel(separate.outputs["Y"]),
            combine.inputs["Y"]
        )

        links.new(
            remap_channel(separate.outputs["Z"]),
            combine.inputs["Z"]
        )

        links.new(
            combine.outputs["Vector"],
            emission.inputs["Color"]
        )

        links.new(
            emission.outputs["Emission"],
            output.inputs["Surface"]
        )

        obj.data.materials.clear()
        obj.data.materials.append(normal_mat)

# ---------------------------------------------------
# NORMAL RENDER
# ---------------------------------------------------

setup_camera_space_normal_materials()

tree = clear_compositor()

render_layers = tree.nodes.new("CompositorNodeRLayers")

camera_normal_out = tree.nodes.new(
    "CompositorNodeOutputFile"
)

camera_normal_out.base_path = OUTPUT_DIR
camera_normal_out.file_slots[0].path = f"normal_camera_"

tree.links.new(
    render_layers.outputs["Image"],
    camera_normal_out.inputs[0]
)

scene.render.filepath = os.path.join(
    OUTPUT_DIR,
    f"normal_camera_direct.png"
)

bpy.ops.render.render(write_still=True)

# ---------------------------------------------------
# METADATA
# ---------------------------------------------------

meta = {
    "frame_id": FRAME_ID,
    "asset": bpy.data.filepath,
    "hdri_path": HDRI_REL_PATH,
    "resolution": [
        scene.render.resolution_x,
        scene.render.resolution_y
    ],
    "samples": scene.cycles.samples,
    "outputs": {
        "beauty": "beauty_*.png",
        "albedo": "albedo_*.png",
        "mask": "mask_*.png",
        "normal_world_debug": "normal_world_*.png",
        "normal_camera": "normal_camera_*.png",
    },
}

with open(
    os.path.join(OUTPUT_DIR, "meta.json"),
    "w"
) as f:
    json.dump(meta, f, indent=2)

print("DONE.")
print("Outputs written to:", OUTPUT_DIR)