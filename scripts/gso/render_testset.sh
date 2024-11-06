obj_id=Sushi_Mat
blender-3.2.2-linux-x64/blender --background --python render.py \
    -- --object_path=${PATH_TO_GSO}/${obj_id}/meshes/model.obj \
    --output_folder=render_outputs/gso/testset/$obj_id \
    --train_elevation_sin_amplitude_max_levels=10-20 \
    --test_num_per_floor=3 \
    --test_elevation_range=70-90 \
    --use_gpu