import bpy
import os
import json
from pathlib import Path

ROOT = "/Users/hrithikg/SWRepos/switchlight"

HDRI_PATH = os.path.join(
    ROOT,
    "data/blender/hdri/studio_small_03_2k.hdr"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "data/blender/smoke_test"
)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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
# HELPER: SET FILE OUTPUT COMPOSITOR
# ---------------------------------------------------

def clear_compositor():
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    return tree


# ---------------------------------------------------
# RENDER 1: BEAUTY / WORLD NORMAL / MASK
# ---------------------------------------------------

tree = clear_compositor()
render_layers = tree.nodes.new("CompositorNodeRLayers")

beauty_out = tree.nodes.new("CompositorNodeOutputFile")
beauty_out.label = "Beauty"
beauty_out.base_path = OUTPUT_DIR
beauty_out.file_slots[0].path = "beauty_"

normal_out = tree.nodes.new("CompositorNodeOutputFile")
normal_out.label = "WorldSpaceNormal"
normal_out.base_path = OUTPUT_DIR
normal_out.file_slots[0].path = "normal_world_"

mask_out = tree.nodes.new("CompositorNodeOutputFile")
mask_out.label = "Mask"
mask_out.base_path = OUTPUT_DIR
mask_out.file_slots[0].path = "mask_"

tree.links.new(render_layers.outputs["Image"], beauty_out.inputs[0])
tree.links.new(render_layers.outputs["Normal"], normal_out.inputs[0])
tree.links.new(render_layers.outputs["Alpha"], mask_out.inputs[0])

scene.render.filepath = os.path.join(OUTPUT_DIR, "beauty_direct.png")
bpy.ops.render.render(write_still=True)

# ---------------------------------------------------
# STORE ORIGINAL MATERIALS
# ---------------------------------------------------

original_materials = {}

for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        original_materials[obj.name] = [slot.material for slot in obj.material_slots]


# ---------------------------------------------------
# RENDER 2: TRUE ALBEDO USING EMISSION OVERRIDE
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
            emission.inputs["Color"].default_value = base_color_input.default_value

        for link in list(output.inputs["Surface"].links):
            links.remove(link)

        links.new(emission.outputs["Emission"], output.inputs["Surface"])

tree = clear_compositor()
render_layers = tree.nodes.new("CompositorNodeRLayers")

albedo_out = tree.nodes.new("CompositorNodeOutputFile")
albedo_out.label = "Albedo"
albedo_out.base_path = OUTPUT_DIR
albedo_out.file_slots[0].path = "albedo_"

tree.links.new(render_layers.outputs["Image"], albedo_out.inputs[0])

scene.render.filepath = os.path.join(OUTPUT_DIR, "albedo_direct.png")
bpy.ops.render.render(write_still=True)


# ---------------------------------------------------
# RESTORE ORIGINAL MATERIALS BEFORE CAMERA NORMAL PASS
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
# CAMERA-SPACE NORMAL MATERIAL OVERRIDE
# ---------------------------------------------------

def setup_camera_space_normal_materials():
    """
    Replaces all mesh materials with a camera-space normal material.

    Output convention:
    RGB stores normal remapped from [-1, 1] to [0, 1].
    Renderer loader should convert back using:
        normal = rgb * 2.0 - 1.0
    """

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue

        normal_mat = bpy.data.materials.new(name=f"{obj.name}_camera_space_normal_mat")
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

        x_mul = nodes.new(type="ShaderNodeMath")
        x_mul.operation = "MULTIPLY"
        x_mul.inputs[1].default_value = 0.5

        x_add = nodes.new(type="ShaderNodeMath")
        x_add.operation = "ADD"
        x_add.inputs[1].default_value = 0.5

        y_mul = nodes.new(type="ShaderNodeMath")
        y_mul.operation = "MULTIPLY"
        y_mul.inputs[1].default_value = 0.5

        y_add = nodes.new(type="ShaderNodeMath")
        y_add.operation = "ADD"
        y_add.inputs[1].default_value = 0.5

        z_mul = nodes.new(type="ShaderNodeMath")
        z_mul.operation = "MULTIPLY"
        z_mul.inputs[1].default_value = 0.5

        z_add = nodes.new(type="ShaderNodeMath")
        z_add.operation = "ADD"
        z_add.inputs[1].default_value = 0.5

        combine = nodes.new(type="ShaderNodeCombineXYZ")

        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0

        output = nodes.new(type="ShaderNodeOutputMaterial")

        links.new(geom.outputs["Normal"], vec_transform.inputs["Vector"])
        links.new(vec_transform.outputs["Vector"], separate.inputs["Vector"])

        links.new(separate.outputs["X"], x_mul.inputs[0])
        links.new(x_mul.outputs[0], x_add.inputs[0])
        links.new(x_add.outputs[0], combine.inputs["X"])

        links.new(separate.outputs["Y"], y_mul.inputs[0])
        links.new(y_mul.outputs[0], y_add.inputs[0])
        links.new(y_add.outputs[0], combine.inputs["Y"])

        links.new(separate.outputs["Z"], z_mul.inputs[0])
        links.new(z_mul.outputs[0], z_add.inputs[0])
        links.new(z_add.outputs[0], combine.inputs["Z"])

        links.new(combine.outputs["Vector"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])

        obj.data.materials.clear()
        obj.data.materials.append(normal_mat)


# ---------------------------------------------------
# RENDER 3: CAMERA-SPACE NORMALS
# ---------------------------------------------------

setup_camera_space_normal_materials()

tree = clear_compositor()
render_layers = tree.nodes.new("CompositorNodeRLayers")

camera_normal_out = tree.nodes.new("CompositorNodeOutputFile")
camera_normal_out.label = "CameraSpaceNormal"
camera_normal_out.base_path = OUTPUT_DIR
camera_normal_out.file_slots[0].path = "normal_camera_"

tree.links.new(render_layers.outputs["Image"], camera_normal_out.inputs[0])

scene.render.filepath = os.path.join(OUTPUT_DIR, "normal_camera_direct.png")
bpy.ops.render.render(write_still=True)

# ---------------------------------------------------
# METADATA
# ---------------------------------------------------

meta = {
    "asset": bpy.data.filepath,
    "hdri_path": HDRI_PATH,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "samples": scene.cycles.samples,
    "outputs": {
        "beauty": "beauty_*.png",
        "albedo": "albedo_*.png",
        "mask": "mask_*.png",
        "normal_world_debug": "normal_world_*.png",
        "normal_camera": "normal_camera_*.png",
    },
    "notes": {
        "beauty": "Cycles beauty render under HDRI lighting, transparent film enabled.",
        "albedo": "Emission override using material base color / base color texture. Intended as lighting-independent diffuse color approximation.",
        "mask": "Render layer alpha with transparent film. White foreground, black background.",
        "normal_world_debug": "Blender default normal pass. World-space debug only. Do not use for renderer training.",
        "normal_camera": "Camera-space normal remapped from [-1,1] to [0,1]. Use rgb * 2 - 1 when loading into torch.",
        "normal_contract": "Expected downstream tensor: [3,H,W], float32, camera-space, unit vectors, +Z toward camera if convention check passes.",
    }
}

with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("DONE.")
print("Outputs written to:", OUTPUT_DIR)