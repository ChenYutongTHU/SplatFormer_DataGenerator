"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector
from copy import deepcopy


IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

import bpy
import os

def load_image_and_check(file_path):
    # Load the image
    img = bpy.data.images.load(file_path)

    # Ensure image data is available
    img.pixels.update()

    # Access the pixel data (RGBA: 4 values per pixel)
    pixels = list(img.pixels)

    # The pixel data is in the form of (R, G, B, A) tuples in a flat list
    # We are interested in checking non-black (non-empty) pixels
    total_pixels = len(pixels) // 4  # each pixel has 4 values (R, G, B, A)

    non_empty_pixel_count = 0

    # Loop through each pixel, checking if it's non-empty (non-black)
    for i in range(0, len(pixels), 4):
        r, g, b, a = pixels[i:i+4]
        if a > 0:  # Check if the pixel is not black
            non_empty_pixel_count += 1

    # Check if more than 10% of pixels are non-empty
    non_empty_ratio = non_empty_pixel_count / total_pixels

    if non_empty_ratio > 0.1:
        print(f"More than 10% of pixels are non-empty: {non_empty_ratio*100:.2f}%")
        return True
    else:
        print(f"Less than 10% of pixels are non-empty: {non_empty_ratio*100:.2f}%")
        return False


def get_rotation(object_file):
    # Rotate some objects to make them face the top camera
    obj_name = os.path.basename(object_file).split('.')[0]
    if obj_name in ['9d47c599aa654957b56706a58cc389c1', '467c6873188044ee85066fa3cea957f5']:
        return  '+y'
    elif obj_name in ['43e13f812b1c448883e5e5674d9d6f39','45a0272235cc4f4a95d18905df3a3972','86a52a28b2534c96bf911ef6e8e3e26c', '590bd63837e74e5093233a18ed091dd9','3f0b63f78c004821bdf73aa7acd2e917','98c079a608fa4088b0960b84da71eb01']:
        return '-x'
    elif obj_name in ['67e039b5470243089c18e919366366df','00e0d74702bb4799b2240ffac69ea931','2ffe996e845d47b1acbdf5034812a411']:
        return '+x'
    elif 'GSO/unextracted' in object_file:
        obj_name = object_file.split('/')[-3]   
        if obj_name in ['Dog','Crosley_Alarm_Clock_Vintage_Metal','SpiderMan_Titan_Hero_12Inch_Action_Figure_oo1qph4wwiW','Olive_Kids_Dinosaur_Land_Pack_n_Snack']:
            return '-x'
        elif obj_name in ['Sootheze_Cold_Therapy_Elephant','ZX700_mf9Pc06uL06','Poppin_File_Sorter_Pink','AMBERLIGHT_UP_W']:
            return '+y'
        elif obj_name in ['Nintendo_Mario_Action_Figure']:
            return [-1*np.pi/2,np.pi/4,0]
        else:
            return 'null'
    else:
        return 'null'


def sample_train_elevation_degree(split, step, total_step, random_offset, elevation_level):
    elevation_degree_sin_frequency = getattr(args, 'train_elevation_sin_frequency')
    elevation_degree_sin_amplitude_min = getattr(args, 'train_elevation_sin_amplitude_min')
    elevation_degree_sin_amplitude_max = elevation_level
    step = (step + random_offset) % total_step
    mid = 0.5*(elevation_degree_sin_amplitude_min + elevation_degree_sin_amplitude_max)
    T = total_step/elevation_degree_sin_frequency
    sin_part = np.sin(2*np.pi*step/T)
    elevation = mid + (elevation_degree_sin_amplitude_max - mid)*sin_part
    return elevation

def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """

    # Set up rendering
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    render.engine = args.engine
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"  # or "OPENCL"
    bpy.context.scene.cycles.samples = 256
    # scene.cycles.filter_width = 0.01
    scene.cycles.use_adaptive_sampling = True
    # scene.cycles.adaptive_threshold = 0.05 (Let cycles decide it automatically)
    scene.cycles.use_denoising = True

    with open('log.txt','w') as f:
        for attribute_name in dir(bpy.context.scene.cycles):
            attribute_value = getattr(bpy.context.scene.cycles, attribute_name)
            f.write(attribute_name + ' = ' + str(attribute_value) + '\n')
    render.image_settings.color_mode = 'RGBA'
    render.image_settings.color_depth = '8'
    render.image_settings.file_format = 'PNG'
    render.resolution_x = args.resolution
    render.resolution_y = args.resolution
    render.resolution_percentage = 100
    render.film_transparent = True


    scene.use_nodes = True
    scene.view_layers["View Layer"].use_pass_normal = False #True
    scene.view_layers["View Layer"].use_pass_diffuse_color = False #True
    scene.view_layers["View Layer"].use_pass_object_index = False #True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')
    # Delete default cube
    context.active_object.select_set(True)
    bpy.ops.object.delete()

    bpy.ops.object.select_all(action='DESELECT')


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    bpy.ops.import_scene.obj(filepath=args.object_path)
    obj = bpy.context.selected_objects[0]

    context.view_layer.objects.active = obj
    obj.cycles_visibility.shadow = False
    obj.cycles.cast_shadow = False
    context.object.cycles_visibility.shadow = False

    # Disable specular shading (for shapenet)
    for slot in obj.material_slots:
        nodes = slot.material.node_tree.nodes
        links = slot.material.node_tree.links
        diffuse = nodes.new( type = 'ShaderNodeBsdfDiffuse' )
        diffuse.inputs['Roughness'].default_value = 0 # Standard lambertian reflection
        if 'Image Texture' in nodes: #use image texture
            image_texture = nodes['Image Texture']
            links.new( image_texture.outputs['Color'], diffuse.inputs['Color'] )
        else: #copy the color
            color = nodes['Principled BSDF'].inputs['Base Color']
            diffuse.inputs['Color'].default_value = color.default_value
        link = links.new( diffuse.outputs['BSDF'], nodes['Material Output'].inputs['Surface'] )

    #For .obj file, we need to reset the rotation of the object 
    # [This is not done in the previous version]
    # for obj in get_scene_root_objects():
    #     obj.rotation_euler = (0,0,0) 
    print('after reset rotation')

def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, matrix: Optional[np.ndarray] = None
) :
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            if matrix is not None:
                coord = [c for c in coord] #(x y z)
                coord = np.array(coord) #(3,)
                matrix_world = np.array([[e for e in vec] for vec in matrix]) #3,3
                coord = matrix_world @ coord #(4,)
                coord = coord[:3] #x y z
            #Note that here the coord is in the original obj's system [Not after the OpenGL's conversion]
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return bbox_min, bbox_max


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.
    #including the cube
    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)

def normalize_scene(rotate) -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty


    if rotate=='null':
        pass
    elif rotate in ['+y','-y']:
        #bbox_min = (bbox_min[2], bbox_min[1], bbox_min[0]) #swap x and z
        #bbox_max = (bbox_max[2], bbox_max[1], bbox_max[0])
        #x_length, z_length = z_length, x_length
        for obj in get_scene_root_objects():
            obj.rotation_euler = (obj.rotation_euler[0], 
                                obj.rotation_euler[1]+np.pi/2 if rotate=='+y' else obj.rotation_euler[1]-np.pi/2,
                                obj.rotation_euler[2])
    elif rotate in ['+x','-x']:
        #bbox_min = (bbox_min[0], bbox_min[2], bbox_min[1]) #swap y and z
        #bbox_max = (bbox_max[0], bbox_max[2], bbox_max[1])
        #y_length, z_length = z_length, y_length
        for obj in get_scene_root_objects():
            #print(obj.rotation_euler)
            obj.rotation_euler = (obj.rotation_euler[0]+np.pi/2 if rotate=='+x' else obj.rotation_euler[0]-np.pi/2,
            obj.rotation_euler[1], obj.rotation_euler[2])
            #print(obj.rotation_euler)
    elif type(rotate) == list:
        for obj in get_scene_root_objects():
            obj.rotation_euler = (rotate[0],rotate[1],rotate[2])
    else:
        raise ValueError(f'Invalid rotate value: {rotate}')

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects(): 
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None
    return obj #memorize the last obj for later use


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def render_object(
) -> None:
    object_file = args.object_path
    os.makedirs(args.output_folder, exist_ok=True)

    reset_scene()
    load_object(object_file)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.modifier_add(type='EDGE_SPLIT')
    context.object.modifiers["EdgeSplit"].split_angle = 1.32645
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")
    obj = bpy.context.selected_objects[0]
    obj.pass_index = 1
    # Make light just directional, disable shadows.
    light = bpy.data.lights['Light']
    light.type = 'SUN' #However, light is stil.bpy.types.PointLight
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()  #Delete the light

    bpy.ops.object.light_add(type='SUN')
    light1 = bpy.data.lights['Sun']
    light1.use_shadow = False
    light1.cycles.cast_shadow = False
    light1.energy = 10
    light1.angle = math.radians(120) #angular diameter (soft shadow)
    bpy.data.objects['Sun'].location = (0, -1, 1.5)
    bpy.data.objects['Sun'].rotation_euler = (0, 0, 0)
    light1.specular_factor = 0

    bbox_min, bbox_max = scene_bbox(obj, matrix = obj.matrix_world.to_3x3()) #include the initial rotation
    x_length = bbox_max[0] - bbox_min[0]
    y_length = bbox_max[1] - bbox_min[1] 
    z_length = bbox_max[2] - bbox_min[2]

    #In shapenet, rotate the object so that the shortest side is along the z axis
    lengths_sorted = sorted([x_length, y_length, z_length])
    if z_length == lengths_sorted[0]:
        pass
    elif y_length == lengths_sorted[0]:
        obj.rotation_euler = (np.pi/2+np.pi/2, 0, 0)  #rotate around x axis (include the initial rotation)
        y_length, z_length = z_length, y_length #swap y and z

    elif x_length == lengths_sorted[0]:
        obj.rotation_euler = (np.pi/2, np.pi/2, 0)  #rotate around y axis
        x_length, z_length = z_length, x_length
    scale = 1/max(x_length, y_length, z_length)
    obj.scale = obj.scale*scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox(obj, matrix = obj.matrix_world.to_3x3())
    bbox_min = np.array(bbox_min)
    bbox_max = np.array(bbox_max)
    offset = -(bbox_min + bbox_max) / 2
    obj.matrix_world.translation = (obj.matrix_world.translation[0] + offset[0], obj.matrix_world.translation[1] + offset[1], obj.matrix_world.translation[2] + offset[2])
    bbox_min = np.array(bbox_min) + offset
    bbox_max = np.array(bbox_max) + offset
    cam_z = 0 
    lengths_sorted = sorted([x_length, y_length, z_length])
    #assert z_length == lengths_sorted[0]
    ratio = lengths_sorted[-1]/lengths_sorted[-2]
    if ratio > 7:
        print("The object is too long, discard")
        exit()

    cam = scene.objects['Camera']
    cam.rotation_euler = (0,0,0)
    cam.data.lens = 100
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, cam_z)
    cam.parent = cam_empty
    scene.collection.objects.link(cam_empty)
    context.view_layer.objects.active = cam_empty
    cam_constraint.target = cam_empty
    rotation_mode = 'XYZ'

    fxy = args.resolution / cam.data.sensor_width * cam.data.lens
    cx = args.resolution / 2
    cy = args.resolution / 2

    '''
    # This is some legacy issue
    '''
    if args.legacy:
        bbox_vertex = np.array([
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
        ])
        right_transform = np.array([[1,0,0],[0,0,-1],[0,1.,0]])
        right_transform_inverse = right_transform.T
        bbox_vertex = bbox_vertex@right_transform_inverse.T #(N,3),
        bbox_min = bbox_vertex.min(axis=0)
        bbox_max = bbox_vertex.max(axis=0)
    

    f_images_lines = {}
    levelsplits = [int(l) for l in args.train_elevation_sin_amplitude_max_levels.split('-')]
    for level_split in levelsplits:
        level_split = str(level_split)
        if not args.generate_trainset:
            os.makedirs(os.path.join(os.path.abspath(args.output_folder),level_split, 'sparse/0'), exist_ok=True)
            os.makedirs(os.path.join(os.path.abspath(args.output_folder),level_split, 'images'), exist_ok=True)
            with open(os.path.join(os.path.abspath(args.output_folder), level_split, 'sparse/0/bbox.txt'), 'w') as f:
                f.write(f'{bbox_min[0]} {bbox_min[1]} {bbox_min[2]}\n')
                f.write(f'{bbox_max[0]} {bbox_max[1]} {bbox_max[2]}\n')
            with open(os.path.join(os.path.abspath(args.output_folder),level_split, 'sparse/0/cameras.txt'), 'w') as f:
                f.writelines(f'1 SIMPLE_PINHOLE {args.resolution} {args.resolution} {fxy} {cx} {cy}\n')
        f_images_lines[level_split] = []
    if args.generate_trainset:
        os.makedirs(os.path.join(os.path.abspath(args.output_folder), 'images'), exist_ok=True)
        os.makedirs(os.path.join(os.path.abspath(args.output_folder), 'sparse/0'), exist_ok=True)
        with open(os.path.join(os.path.abspath(args.output_folder), 'sparse/0/bbox.txt'), 'w') as f:
            f.write(f'{bbox_min[0]} {bbox_min[1]} {bbox_min[2]}\n')
            f.write(f'{bbox_max[0]} {bbox_max[1]} {bbox_max[2]}\n')
        with open(os.path.join(os.path.abspath(args.output_folder), 'sparse/0/cameras.txt'), 'w') as f:
            f.writelines(f'1 SIMPLE_PINHOLE {args.resolution} {args.resolution} {fxy} {cx} {cy}\n')
    line_id = 1
    
    random.seed(ord(os.path.basename(object_file).split('.')[0][-1]))
    random_offset = np.random.randint(0, args.train_elevation_sin_frequency)

    distance = args.distance

    for split in ['train_'+str(l) for l in levelsplits]+['test']:
        if split == 'test':
            #views = getattr(args, f'{split}_views')
            ele_start, ele_end = args.test_elevation_range.split('-')
            elevations = np.arange(int(ele_start), int(ele_end)+1, 10)
            views = args.test_num_per_floor*len(elevations)
            stepsize = 360.0 / args.test_num_per_floor
        else:
            views = getattr(args, 'train_views')
            stepsize = 360.0 / views
        if views==0:
            continue
        if 'train' in split:
            train_distances = []
            bbox_min, bbox_max = scene_bbox()
            centerx = (bbox_max[0] + bbox_min[0]) / 2
            centery = (bbox_max[1] + bbox_min[1]) / 2
            cam_empty.location = (centerx, centery, cam_z)

        for i in range(0, views):
            cam.location = (0,distance, 0) #precomputed
            
            fp = os.path.join(os.path.abspath(args.output_folder),  'images', split)
            render_file_path = fp + '_{0:03d}'.format(int(i))
            if split=='test':
                elevation_degree = elevations[i%len(elevations)]
            else:
                elevation_degree = sample_train_elevation_degree(split, i, views, random_offset, elevation_level=int(split.split('_')[-1]))
            cam_empty.rotation_euler[0] = math.radians(elevation_degree if elevation_degree<90 else 89)

            scene.render.filepath = render_file_path
            bpy.ops.render.render(write_still=True)  # render still
            RT = get_3x4_RT_matrix_from_blender(cam)
            W2C_OPENGL = np.eye(4)
            W2C_OPENGL[:3,:4] = RT#[0]
            left_transform = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])  # the camera's coordinate
            if args.legacy:
                right_transform = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]) # the world's coordinate
                W2C_COLMAP = (left_transform@W2C_OPENGL)@right_transform
            else:
                W2C_COLMAP = left_transform@W2C_OPENGL
            # """
            # The reconstructed pose of an image is specified as the projection from world to the camera coordinate system of an image 
            # using a quaternion (QW, QX, QY, QZ) and a translation vector (TX, TY, TZ). 
            # """
            R, T = W2C_COLMAP[:3,:3], W2C_COLMAP[:3,3] #3x3, 3
            qvec = rotmat2qvec(R)
            #f_images.write(f'{line_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} 1 {os.path.basename(render_file_path)+".png"}\n') #1 is the camera id
            #f_images.write('\n')
            if 'train' in split:
                if args.generate_trainset:
                    new_path = os.path.join(os.path.abspath(args.output_folder),'images',f'train_{int(i):03d}.png')
                else:
                    new_path = os.path.join(os.path.abspath(args.output_folder),split.split('_')[-1],'images',f'train_{int(i):03d}.png')
                os.system(f'mv {render_file_path}.png {new_path}')
                line_id = len(f_images_lines[split.split('_')[-1]])//2
                f_images_lines[split.split('_')[-1]].append(f'{line_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} 1 {os.path.basename(new_path)}\n')
                f_images_lines[split.split('_')[-1]].append('\n')
            elif 'test' == split:
                for level_split in levelsplits:
                    level_split = str(level_split)
                    if args.generate_trainset:
                        new_path_cp = os.path.join(os.path.abspath(args.output_folder),'images',f'test_elevation{elevation_degree}_step{i//len(elevations)}.png')
                        os.system(f'mv {render_file_path}.png {new_path_cp}')
                    else:
                        new_path_cp = os.path.join(os.path.abspath(args.output_folder),level_split,'images',f'test_elevation{elevation_degree}_step{i//len(elevations)}.png')
                        os.system(f'cp {render_file_path}.png {new_path_cp}')
                    line_id = len(f_images_lines[level_split])//2
                    f_images_lines[level_split].append(f'{line_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} 1 {os.path.basename(new_path_cp)}\n')
                    f_images_lines[level_split].append('\n')
            else:
                raise ValueError(f"Unknown split {split}")
            

            C2W_OPENGL = np.linalg.inv(W2C_OPENGL)
            cam_xyz = C2W_OPENGL[:3,3]
            if split!='test':
                cam_empty.rotation_euler[2] += math.radians(stepsize) #for the next view
            elif split=='test' and i%len(elevations)==len(elevations)-1:
                cam_empty.rotation_euler[2] += math.radians(stepsize)
            #read the first image
            if True and i==0 and 'train' in split:
                if args.generate_trainset:
                    file_path = os.path.join(os.path.abspath(args.output_folder),'images','train_000.png')
                else:
                    file_path = os.path.join(os.path.abspath(args.output_folder),split.split('_')[-1],'images','train_000.png')
                if load_image_and_check(file_path)==False:
                    import shutil
                    print(args.output_folder)
                    shutil.rmtree(os.path.abspath(args.output_folder))
                    exit()

    if not args.generate_trainset:
        for key, lines in f_images_lines.items():
            with open(os.path.join(os.path.abspath(args.output_folder),key,'sparse/0/images.txt'), 'w') as f:
                for line in lines:
                    f.write(line)
        os.system(f'rm -rf {os.path.join(os.path.abspath(args.output_folder), "images")}')
    else:
        assert len(f_images_lines) == 1
        for key, lines in f_images_lines.items():
            with open(os.path.join(os.path.abspath(args.output_folder), 'sparse/0', 'images.txt'), 'w') as f:
                for line in lines:
                    f.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="CYCLES",
        choices=["CYCLES", "BLENDER_EEVEE"],
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution of the rendered images.",
    )
    parser.add_argument(
        "--train_views",
        type=int,
        default=32,
    )
    # parser.add_argument(
    #     "--test_views",
    #     type=int,
    #     default=9,
    # )

    parser.add_argument(
        "--train_elevation_sin_frequency",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--train_elevation_sin_amplitude_min",
        type=int,
        default=0,
    )
    
    # parser.add_argument(
    #     "--train_elevation_sin_amplitude_max",
    #     type=int,
    #     default=15,
    # )

    parser.add_argument(
        '--train_elevation_sin_amplitude_max_levels',
        type=str, required=True,
        #default='15,18,21'
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=3.5,
    )

    parser.add_argument(
        "--test_num_per_floor",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--test_elevation_range",
        type=str,
        default='20-90'
    )

    parser.add_argument(
        "--generate_trainset",
        action='store_true'
    )

    parser.add_argument(
        "--legacy",
        action='store_true' #For shapenet
    )
    
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set render settings
    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = args.resolution
    render.resolution_y = args.resolution
    render.resolution_percentage = 100

    # Set cycles settings
    scene.cycles.device = "GPU"
    if args.object_path.endswith(".blend"):
        scene.cycles.samples = 1024
    else:
        scene.cycles.samples = 256
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"  # or "OPENCL"

    # Render the images
    render_object()
