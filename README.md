# *Fairness-Aware Class Imbalanced Learning on Multiple Subgroups*

Official Implementation of our paper 
***Fairness-Aware Class Imbalanced 
Learning on Multiple Subgroups*** (Submitted to UAI 2023)  


## Requirements

### ENVS

The algorithm is implemented mainly based on PyTorch Deep Learning Framework. 
To install the related packages, use
```bash
pip install -r requirements.txt
```

### Data

- Datasets
  - Alzheimer’s Disease (Tadpole)
  - Credit Card (Credit)

---

## Getting Started - Train

1. Use our provided data files and Make Sure that there are well-prepared representative data in the `./DATASOURCE/#your_dataset` folder.

2. Use `all_train.py` script to train your model.
   1. Use config file in `./EXPS` to train your model.

        ```cmd
        python all_train.py -config EXPS/amazon_template.yml
        ```

        We have provided our template configs on *Tadpole* and *Credit* datasets.

   2. Directly passing the changeable parameters to the training script as:

        ```cmd
        python all_train.py --method ours --dataset amazon --N_subtask 10
        ```

    Note: The config file has higher priority than the direct passed params.

3. Check the result:

   1. Log file will be output to `./logs` folder
   2. Numerical test result will be saved to `./npy` folder  

   