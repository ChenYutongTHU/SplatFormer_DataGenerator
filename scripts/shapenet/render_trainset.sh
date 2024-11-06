PATH_TO_SHAPENET='' #Replace with the path to the ShapeNetCore.v2 directory
obj_id='03046257/8c00d87bcc8f034aa1b95b258b5f7139' #Replace other scenes in OOD-BlenderRender/traintest_splits/shapenet_train.txt
blender-2.90.0-linux64/blender --background --python render_shapenet.py  \
    -- --object_path=${PATH_TO_SHAPENET}/${obj_id}/models/model_normalized.obj \
    --output_folder=render_outputs/shapenet/trainset/${obj_id} \
    --train_elevation_sin_amplitude_max_levels=10 \
    --test_num_per_floor=2 \
    --test_elevation_range=70-90 \
    --generate_trainset  --use_gpu