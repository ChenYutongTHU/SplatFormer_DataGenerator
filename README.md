# SplatFormer-OOD-BlenderRender
This repository is used by [SplatFormer]() to generate OOD data. It contains 
* The blender rendering scripts to render OOD-NVS test and training set. 
* The nerfstudio script to train 3DGS on the rendered OOD-NVS sets.
You can also download **the prepared test set** [here](). 

## 1. Installation
### Blender (for render)
```
wget https://download.blender.org/release/Blender2.90/blender-2.90.0-linux64.tar.xz # For ShapeNet
tar -xvf blender-2.90.0-linux64.tar.xz
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz # For Objaverse-v1 and GSO
tar -xvf blender-3.2.2-linux-x64
```
### nerfstudio (for 3DGS training)
```
cd nerfstudio
pip install -e .
```

## 2. Rendering 3D Scenes
### ShapeNet
#### Download
Download ShapeNet-Core from [the huggingface repository](https://huggingface.co/datasets/ShapeNet/ShapeNetCore). Or directly
```
## In Python
from huggingface_hub import snapshot_download
download_dir = 'path/to/the/dir'
snapshot_download(repo_id="ShapeNet/ShapeNetCore",repo_type="dataset",local_dir=download_dir, cache_dir=download_dir)
```
We select ShapeNet objects with texture for OOD training and test. The training (29,587 objects) and test (20 objects) split can be found in [](traintest_splits).
#### Render the training set
To render each object for OOD training and put the camera poses and images in a colmap structure
```
obj_id='03046257/8c00d87bcc8f034aa1b95b258b5f7139' #Replace other scenes in OOD-BlenderRender/traintest_splits/shapenet_train.txt
blender-2.90.0-linux64/blender --background --python render_shapenet.py  \
    -- --object_path=${PATH_TO_SHAPENET}/${obj_id}/models/model_normalized.obj \
    --output_folder=render_outputs/shapenet/trainset/${obj_id} \
    --train_elevation_sin_amplitude_max_levels=10 \
    --test_num_per_floor=2 \
    --test_elevation_range=70-90 \
    --generate_trainset  --use_gpu
```
#### Render the test set
For OOD test sets, we render each object with two sets of elevation ranges, 10 and 20. We also render more test views with various elevation degree.
```
obj_id='02691156/2628b6cfcf1a53465569af4484881d20' #Replace other scenes in OOD-BlenderRender/traintest_splits/shapenet_test.txt
blender-2.90.0-linux64/blender --background --python render_shapenet.py  \
    -- --object_path=${PATH_TO_SHAPENET}/${obj_id}/models/model_normalized.obj \
    --output_folder=render_outputs/shapenet/trainset/${obj_id} \
    --train_elevation_sin_amplitude_max_levels=10,20 \
    --test_num_per_floor=3 \
    --test_elevation_range=20-90  \
    --use_gpu
```
In SplatFormer, we use only 48k scenes for training. The training (48,354 objects) and test (20 objects) split can be found in [traintest_splits](traintest_splits).
### Objaverse-v1
#### Download
Download objaverse-v1 from [the huggingface repository](https://huggingface.co/datasets/allenai/objaverse/tree/main). For the OOD training set, we used [000-000,...,000-039]. For the OOD evaluation set, we choose objects from [000-100](https://huggingface.co/datasets/allenai/objaverse/tree/main/glbs/000-100). To download these subsets, you can use [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
```
# Take 000-100 for examples
huggingface-cli download allenai/objaverse \
--include "glbs/000-100/*.glb" \
--repo-type dataset \
--token='your-huggging-face-token'  \
--use_gpu
```

#### Render training set
```
obj_id='fff5b37e11e74f00a6459b37451950b4'
blender-3.2.2-linux-x64/blender --background --python render.py \
    -- --object_path=${PATH_TO_OBJAVERSE}/$obj_id.glb \
    --output_folder=render_outputs/objaverse/trainset/$obj_id \
    --train_elevation_sin_amplitude_max_levels=15 \
    --test_num_per_floor=2 \
    --test_elevation_range=70-90 \
    --generate_trainset  \
    --use_gpu
```
#### Render Test set
```
obj_id='0a604c1ee9b245c7b2d797a910e53219'
blender-3.2.2-linux-x64/blender --background --python render.py \
    -- --object_path=${PATH_TO_OBJAVERSE}/$obj_id.glb \
    --output_folder=render_outputs/objaverse/testset/$obj_id \
    --train_views=32 \
    --train_elevation_sin_amplitude_max_levels=10-20 \
    --test_num_per_floor=3 \
    --test_elevation_range=70-90  \
    --use_gpu 
```

### Google Scanned Objects
#### Download
Download [google scanned objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) and extract the files. We choose 20 objects and their names are listed in [traintest_splits/gso_test.txt](traintest_splits/gso_test.txt). You can also download them [here]().

#### Render Test set
```
obj_id=Sushi_Mat
blender-3.2.2-linux-x64/blender --background --python render.py \
    -- --object_path=${PATH_TO_GSO}/${obj_id}/meshes/model.obj \
    --output_folder=render_outputs/gso/testset/$obj_id \
    --train_elevation_sin_amplitude_max_levels=10-20 \
    --test_num_per_floor=3 \
    --test_elevation_range=70-90 \
    --use_gpu
```

## 3. Train 3DGS on the OOD scenes
All the nerfstudio training scripts can be found in [scripts](scripts/). Here we take 3DGS of GSO-OOD test set for example.
```
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
```
Note that when generating the input 3DGS for SplatFormer, we stop the optimization at the 10k step (--early_stop_steps=10000), and reduce sh_degree to save computational budgets. We find that this does not affect 3DGS's rendering performance in these datasets. However, we report the metrics of 3DGS with default configuration (--max_num_iterations=30000, --pipeline.model.sh_degree=3) in our paper's table.
