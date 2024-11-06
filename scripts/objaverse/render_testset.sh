obj_id='0a604c1ee9b245c7b2d797a910e53219'
blender-3.2.2-linux-x64/blender --background --python render.py \
    -- --object_path=${PATH_TO_OBJAVERSE}/$obj_id.glb \
    --output_folder=render_outputs/objaverse/testset/$obj_id \
    --train_views=32 \
    --train_elevation_sin_amplitude_max_levels=10-20 \
    --test_num_per_floor=3 \
    --test_elevation_range=70-90  \
    --use_gpu 