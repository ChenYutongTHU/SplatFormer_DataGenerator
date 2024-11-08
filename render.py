"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

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


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 2.2,
    minz: float = -2.2,
    only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """

    x, y, z = _sample_spherical(
        radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
    )
    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


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


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def randomize_lighting(energy=[5,2,3,2],specular_factor=1, randomize=False) -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    light_dict  = {}
    # Create key light
    if energy[0] > 0:
        light_dict['key_light'] = _create_light(
            name="Key_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(0.785398, 0, -0.785398),
            energy=random.choice([5, 6, 7]) if randomize else energy[0],
            specular_factor=specular_factor,
        )


    # Create fill light
    if energy[1] > 0:
        light_dict['fill_light'] = _create_light(
            name="Fill_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(0.785398, 0, 2.35619),
            energy=random.choice([2, 3, 4]) if randomize else energy[1],
            specular_factor=specular_factor,
        )

    if energy[2] > 0:
        # Create rim light
        light_dict['rim_light'] = _create_light(
            name="Rim_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(-0.785398, 0, -3.92699),
            energy=random.choice([1, 2, 3]) if randomize else energy[2],
            specular_factor=specular_factor,
        )

    if energy[3] > 0:   
        # Create bottom light
        light_dict['bottom_light'] = _create_light(
            name="Bottom_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(3.14159, 0, 0),
            energy=random.choice([1, 2, 3]) if randomize else energy[3],
            specular_factor=specular_factor,
        )
    return dict(
        **light_dict
    )


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)
    print(bpy.ops.file.report_missing_files())
    print(os.path.join(object_path, '../..'))
    print(bpy.ops.file.find_missing_files(directory=os.path.dirname(os.path.dirname(object_path))))
    print('after find missing files')
    print(bpy.ops.file.report_missing_files())
    if file_extension == 'obj':
        #For .obj file, we need to reset the rotation of the object
        for obj in get_scene_root_objects():
            obj.rotation_euler = (0,0,0) 
        print('after reset rotation')

def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, 
    force_matrix: Optional[Matrix] = None,
    ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
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
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            if force_matrix is not None:
                coord = force_matrix @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


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


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }

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

    # load the object
    if object_file.endswith(".blend"):
        bpy.ops.object.mode_set(mode="OBJECT")
        # reset_cameras()
        # delete_invisible_objects()
        reset_scene()
        file_path = object_file
        inner_path = 'Collection'

        bpy.ops.wm.append(filepath=os.path.join(file_path, inner_path, object_name),
        directory=os.path.join(file_path, inner_path),
        filename=object_name)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in scene.objects:
            print(obj)
        print(bpy.ops.file.report_missing_files())
        # exit()

    else:
        reset_scene()
        print(object_file, os.path.isfile(object_file))
        load_object(object_file)

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    metadata_extractor = MetadataExtractor(
        object_path=object_file, scene=scene, bdata=bpy.data
    )
    metadata = metadata_extractor.get_metadata()

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz"):
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        missing_textures = delete_missing_textures()
    metadata["missing_textures"] = missing_textures

    # possibly apply a random color to all objects
    if object_file.endswith(".stl") or object_file.endswith(".ply"):
        assert len(bpy.context.selected_objects) == 1
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None

    # save metadata
    metadata_path = os.path.join(args.output_folder, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)


    # normalize the scene
    rotate = get_rotation(object_file)
    obj_ = normalize_scene(rotate = rotate)
    print(obj_.rotation_euler)
    bbox_min, bbox_max = scene_bbox()
    x_length = bbox_max[0] - bbox_min[0]
    y_length = bbox_max[1] - bbox_min[1] 
    z_length = bbox_max[2] - bbox_min[2]

    #Decide cam_z
    if 'Poppin_File_Sorter_Pink' in object_file:
        cam_z = -0.1
    else:
        cam_z = 0
    lengths_sorted = sorted([x_length, y_length, z_length])
    # assert z_length == lengths_sorted[0]
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

    # randomize the lighting
    if 'GSO/' in object_file:
        energy = [2,2,2,0.3] #Do it deterministically
        randomize_lighting(energy, specular_factor=0.1, randomize=args.generate_trainset)
    else:
        energy = [5,2,3,2]
        randomize_lighting(energy, randomize=args.generate_trainset)
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
    if os.path.basename(object_file).split('.')[0] == '40b3fcc83e354193906ef2d05a48a2e7':
        distance = 4.0
    elif os.path.basename(object_file) == 'chair.blend':
        distance = 4.0
    elif 'Granimals_20_Wooden_ABC_Blocks_Wagon_85VdSftGsLi' in object_file or 'Whey_Protein_Chocolate_12_Packets' in object_file:
        distance = 5.0

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
            # #right_transform = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]) # the world's coordinate
            W2C_COLMAP = (left_transform@W2C_OPENGL)#@right_transform
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
        default=3,
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
        "--use_gpu",
        action='store_true'
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
    scene.cycles.device = "GPU" if args.use_gpu else "CPU"
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
