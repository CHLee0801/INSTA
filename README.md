Instruction Matters, a Simple yet Effective Task Selection Approach in Instruction Tuning for Specific Tasks

This is the official github repository for 'Instruction Matters, a Simple yet Effective Task Selection Approach in Instruction Tuning for Specific Tasks' [[EMNLP 2024](https://arxiv.org/abs/2404.16418)].

Citation:
```
@misc{lee2024instructionmatterssimpleeffective,
      title={Instruction Matters, a Simple yet Effective Task Selection Approach in Instruction Tuning for Specific Tasks}, 
      author={Changho Lee and Janghoon Han and Seonghyeon Ye and Stanley Jungkyu Choi and Honglak Lee and Kyunghoon Bae},
      year={2024},
      eprint={2404.16418},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.16418}, 
}
```


## 0. Install Dependencies
```
conda create -n insta python=3.10
conda activate insta
```

```
# install torch with the correct cuda version, check nvcc --version
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# install Hugging Face Libraries
pip install "transformers==4.37.0" "datasets==2.19.1" "accelerate==0.25.0" "evaluate==0.4.0" --upgrade
# install deepspeed and ninja for jit compilations of kernels
pip install "deepspeed==0.9.3" ninja --upgrade
# install additional dependencies needed for training
pip install rouge-score nltk py7zr tensorboard scikit-learn
pip install sentencepiece
pip install wandb
pip install absl-py
```

## 1. Download Data

```
gdown https://drive.google.com/uc?id=1W8tXUZFK-J09kYV1-ZxDOE6QDOrYh_6x
jar xvf data.zip
```

## 2. Train any LMs in Huggingface

```
bash run.sh
```

## 3. Evaluate any LMs in Huggingface

Run the inference.sh file to evaluate! You can either choose task cluster(s) to evaluate or specific task(s) to evaluate.

```
bash inference.sh
```
