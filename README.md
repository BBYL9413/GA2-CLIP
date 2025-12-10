Anonymization processing, retaining only training and testing!!


## Training
This implementation supports Multi-GPU `DistributedDataParallel` training, which is faster and simpler than `DataParallel` used in [ActionCLIP](https://github.com/sallymmx/actionclip). 

- **Single Machine**: To train our model on Kinetics-400 with 8 GPUs in *Single Machine*, you can run:
```sh
# For example, train the 8 Frames ViT-B/32.
bash scripts/run_train.sh  configs/k400/k400_base2novel_f8.yaml
```


#### Testing
```sh
# Single view evaluation. e.g., ViT-B/32 8 Frames on Kinetics-400
bash scripts/run_test.sh  configs/k400/k400_base2novel_f8.yaml exp_nce/k400/ViT-B/16/20251210_121849/model_best.pt


