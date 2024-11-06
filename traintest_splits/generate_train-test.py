import os
'''
#shapenet
with open('/nfs/tang.scratch.inf.ethz.ch/export/tang/cluster/yutongchen/code/nerfstudio/for_Shapenetcore/3d-denoiser/all.seen-categories_train.txt','r') as f:
    shapenet_train = f.readlines()
train_assets  = []
for line in shapenet_train:
    line = line.strip()
    scene = '/'.join(line.split('/')[-2].split('-'))
    train_assets.append(scene)
with open('shapenet_train.txt','w') as f:
    for t in sorted(train_assets):
        f.write(t+'\n')
    
with open('/nfs/tang.scratch.inf.ethz.ch/export/tang/cluster/yutongchen/data/OOD-benchmark/shapenetOOD/test20x1.txt','r') as f:
    shapenet_test = f.readlines()
test_assets  = []
for line in shapenet_test:
    line = line.strip()
    scene = '/'.join(line.split('-')[:-1])
    test_assets.append(scene)
with open('shapenet_test.txt','w') as f:
    for t in sorted(test_assets):
        f.write(t+'\n')
'''

#objaverse
objaverse_test = list(set([scene.split('-')[0] for scene in os.listdir('../../test-set/objaverseOOD/colmap')]))
with open('objaverse_test.txt','w') as f:
    for scene in sorted(objaverse_test):
        f.write(scene+'\n')
# objaverse_train = []
# for line in open('/nfs/tang.scratch.inf.ethz.ch/export/tang/cluster/yutongchen/code/nerfstudio/for_Objaverse/3d-denoiser/snapshot_list_Aug24-a/train.txt','r').readlines():
#     objaverse_train.append(line.strip().split('/')[-2])
# for scene in sorted(objaverse_train):
#     with open('objaverse_train.txt','w') as f:
#         f.write(scene+'\n')

    
