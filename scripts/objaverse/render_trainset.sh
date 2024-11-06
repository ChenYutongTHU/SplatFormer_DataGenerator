obj_id='fff5b37e11e74f00a6459b37451950b4'
blender-3.2.2-linux-x64/blender --background --python render.py \
    -- --object_path=${PATH_TO_OBJAVERSE}/$obj_id.glb \
    --output_folder=render_outputs/objaverse/trainset/$obj_id \
    --train_elevation_sin_amplitude_max_levels=15 \
    --test_num_per_floor=2 \
    --test_elevation_range=70-90 \
    --generate_trainset  \
    --use_gpu