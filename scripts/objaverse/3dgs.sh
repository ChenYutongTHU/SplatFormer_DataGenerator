# In nerfstudio
export CUDA_VISIBLE_DEVICES=0 
for obj in $(ls render_outputs/objaverse/testset)
do
colmap_dir=render_outputs/objaverse/testset/$obj
output_dir=nerfstudio_outputs/objaverse/testset/$obj
ns-train splatfacto \
        --logging.local-writer.enable=False --logging.profiler=none \
        --pipeline.datamanager.data=${colmap_dir} \
        --pipeline.model.sh_degree=1 \
        --pipeline.save_img=True --test_after_train False \
        --output_dir=./ --experiment-name=${output_dir} \
        --relative-model-dir=nerfstudio_models  --vis wandb \
        --steps_per_eval_image=100000 --steps_per_eval_all_images=1000000 --max_num_iterations=30000 \
        --save_only_latest_checkpoint False  --steps_per_save=100000 --save_last_checkpoint True \
        --early_stop_steps=10000 \
        --save_only_gs_params True \
        colmap \
        --downscale_factor=1 \
        --load_3D_points True \
        --auto_scale_poses=False --orientation_method=none --center_method=none \
        --load_bbox True --num_points_from_bbox 50000 \
        --assume_colmap_world_coordinate_convention False \
        --eval_mode filename
done