## Team name: OceanGuardians

### Team members:
- Palásti András (IDNGIS)
- Kurcsi Norbert (Y3ZTEI)
- Wittmajer Dávid (VWXUD6)

## Milestone 1.

For the first milestone we took a closer look on the provided raw 
dataset and created our own from it.

The creation of our own dataset is in: notebooks/data_prep.ipynb

The exploration of our own dataset is in: notebooks/data_exploration.ipynb

## Milestone 2.

### How did we train the model?

  First we used colab for training but our model wasn't nearly as fast that
  it could complete one reasonable training and not run out of colabs limit.
  So we created a training script inside `src/train.py` and used google cloud
  to train the model. Than we integrated automatic mixed precision for our
  training and we doubled our speed with that, so went back to colab.

  For training in colab we used the `notebooks/colab_training.ipynb` notebook.

  For training inside gcp we used the command below:
  ```
  $ PYTHONPATH=. python src/train.py \
      --epochs 5 \
      --batch-size 20 \
      --learning-rate 0.001 \
      --amp
  ```

### How did we evaluate the model?

  Well we used *wandb* for logging and that's a really good for getting a rough
  sense of how the model performs, but also created a script for evaluation
  on test data.

  ```
  $ PYTHONPATH=. python src/train.py \
      --load <pth file of model weights> \
      --eval
  ```

Also our wandb runs are public, so everybody can see our model's training progress,
[our best run yet](https://wandb.ai/andraspalasti2/U-Net/runs/uyxwp30x). 
This has been achieved in colab loading a previous run's weights and continuing
training on it.

**For predicting a single image, use the command below:**
```
$ PYTHONPATH=. python src/train.py \
  --load <pth file of model weights> \
  --predict <image to perform prediction on>
```

### Our article of the whole process: [article](https://github.com/andraspalasti/deeplearning-hw/raw/main/report.pdf)
Weights of the trained models can be found in: 
https://drive.google.com/drive/folders/1pyKEO6eHIgv-YukRxBChiyY3TO8eFsKz?usp=sharing
