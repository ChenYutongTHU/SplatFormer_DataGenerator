PATH_TO_SHAPENET='' #Replace with the path to the ShapeNetCore.v2 directory
obj_id='02691156/2628b6cfcf1a53465569af4484881d20' #Replace other scenes in OOD-BlenderRender/traintest_splits/shapenet_test.txt
blender-2.90.0-linux64/blender --background --python render_shapenet.py  \
    -- --object_path=${PATH_TO_SHAPENET}/${obj_id}/models/model_normalized.obj \
    --output_folder=render_outputs/shapenet/trainset/${obj_id} \
    --train_elevation_sin_amplitude_max_levels=10,20 \
    --test_num_per_floor=3 \
    --test_elevation_range=20-90  \
    --use_gpu