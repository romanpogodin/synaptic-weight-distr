# Synaptic Weight Distributions Depend on the Geometry of Plasticity
Paper: http://arxiv.org/abs/2305.19394

# Setup (with conda)
To install using conda, follow the following steps:

```
conda create -n your_venv python=3.9 conda conda-libmamba-solver -c conda-forge
conda activate your_venv
export CONDA_EXE="$(hash -r; which conda)"
conda config --set solver libmamba
conda install cupy pkg-config compilers libjpeg-turbo opencv pytorch=1.10.2 torchvision=0.11.3 cudatoolkit=11.3 numba terminaltables matplotlib scikit-learn pandas assertpy pytz -c pytorch -c conda-forge
pip install ffcv==0.0.3
pip install setuptools==59.5.0 --force 
```

# ImageNet processing for FFCV
See `./make_ffcv_imagenet.sh`

# Linear regression

`$FOLDER` should be the path where you want the data to be saved. 

```
python ./linear_regression.py --data.results_folder="$FOLDER" --training.corr_scale=1
```

# Finetuning

`$FOLDER` should be the path where you want the data to be saved. 

`$POTENTIAL`: '2-norm', '3-norm' or 'negative_entropy'

`$ND`: 0.5 or 0.75

```
python ./finetuning.py \
    --data.dataset_path=./imagenet/val_500_0.50_90.ffcv \
    --data.num_workers=4 --data.in_memory=1 \
    --data.results_folder="./$FOLDER" \
    --training.potential=$POTENTIAL \
    --training.n_d_dependency=$ND \
    --training.xent_penalty=0 \
    --training.pretrained=1
```
## Processing (for plots)

```
python ./finetuning_processing.py \
    --data.data_path="./$FOLDER" \
    --data.results_folder="./${FOLDER}_preprocessed" \
    --data.pretrained=1
```

# Plots
See `./plots.py`
