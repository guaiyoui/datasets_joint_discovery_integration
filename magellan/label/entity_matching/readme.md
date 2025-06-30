# Entity Matching Using HireGAT

This project is based on [HierGAT](https://github.com/CGCL-codes/HierGAT) and demonstrates how to run the model on our new dataset.

## Step 1: Data Preparation

### Raw Data files

The following raw data files are required:
- `train.csv`
- `test.csv`
- `validate.csv`

### Data Preprocessing

To generate the processed files for HierGAT, run `HierGAT_data_preparation.ipynb`. And will output the following processed files:

- `train_sample_hiergat.txt`
- `test_sample_hiergat.txt`
- `validate_sample_hiergat.txt`

## Step 2: Create Configuration File

Use the following Python code to generate `task_Magellan.json`:

```
import json

task_config = [
    {
        "name": "Magellan",
        "trainset": "newData/train_sample_hiergat.txt",
        "validset": "newData/validate_sample_hiergat.txt",
        "testset": "newData/validate_sample_hiergat.txt",
        "category": [0, 1]
    }
]

with open("task_Magellan.json", "w") as f:
    json.dump(task_config, f, indent=2)
```

## Step 3: Training

Before training, update the `train.py` by setting the configuration to use _"Magellan"_ as the task name.

Then run the following command to start training:

```
python train.py \
  --task Magellan \
  --batch_size 4 \
  --n_epochs 20 \
  --finetuning \
  --save_model
```

## Result

