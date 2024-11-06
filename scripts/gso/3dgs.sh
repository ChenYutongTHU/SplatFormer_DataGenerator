export CUDA_VISIBLE_DEVICES=0 
for obj in $(ls render_outputs/shapenet/testset) # You can modify this to enable multi-gpu processing.
do
colmap_dir=render_outputs/gso/testset/$obj # path to the directory of rendered images
output_dir=nerfstudio_outputs/gso/testset/$obj
ns-train splatfacto \
        --logging.local-writer.enable=False --logging.profiler=none \
        --pipeline.datamanager.data=${colmap_dir} \
        --pipeline.model.sh_degree=1 \
        --pipeline.save_img=True --test_after_train True \
        --output_dir=./ --experiment-name=${output_dir} \
        --relative-model-dir=nerfstudio_models  --vis wandb \
        --max_num_iterations=10000 \
        colmap \
        --downscale_factor=1 \
        --load_3D_points True --load_bbox True --num_points_from_bbox 50000 \
        --auto_scale_poses=False --orientation_method=none --center_method=none \
        --assume_colmap_world_coordinate_convention False \
        --eval_mode filename
done