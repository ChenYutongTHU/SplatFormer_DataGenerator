# In nerfstudio
export CUDA_VISIBLE_DEVICES=0 
for obj in $(ls render_outputs/shapenet/testset)
do
colmap_dir=render_outputs/shapenet/testset/$obj
output_dir=nerfstudio_outputs/shapenet/testset/$obj
ns-train splatfacto \
        --logging.local-writer.enable=False --logging.profiler=none \
        --pipeline.datamanager.data=${colmap_dir} \
        --pipeline.model.sh_degree=0 \
        --pipeline.save_img=True --test_after_train True \
        --output_dir=./ --experiment-name=${output_dir} \
        --relative-model-dir=nerfstudio_models  --vis wandb \
        --save_only_latest_checkpoint False  --steps_per_save=100000 --save_last_checkpoint True \
        --max_num_iterations=10000 \
        --save_only_gs_params True \
        colmap \
        --downscale_factor=1 \
        --load_3D_points True \
        --auto_scale_poses=False --orientation_method=none --center_method=none \
        --load_bbox True --num_points_from_bbox 50000 \
        --assume_colmap_world_coordinate_convention False \
        --eval_mode filename
done