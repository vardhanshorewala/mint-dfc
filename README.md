This repository contains the code and models for our paper:

\textbf{MINT-DFC: Meta-Learning In-Context with Transformers Across Diverse Mixed Function Classes}

![](setting.jpg)

## References

This work builds upon:

**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
_Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant_ <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>

```bibtex
    @InProceedings{garg2022what,
        title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
        author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
        year={2022},
        booktitle={arXiv preprint}
    }
```

## Getting started

You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

   ```
   conda env create -f environment.yml
   conda activate mint-dfc
   ```

2. Download [model checkpoints](https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip) and extract them in the current directory.

   ```
   wget https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip
   unzip models.zip
   ```

3. [Optional] If you plan to train, populate `conf/wandb.yaml` with you wandb info.

That's it! You can now explore our pre-trained models or train your own. The key entry points
are as follows (starting from `src`):

- The `eval.ipynb` notebook contains code to load our own pre-trained models, plot the pre-computed metrics, and evaluate them on new data.
- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. You can try `python train.py --config conf/mixed_function.yaml` for a quick training run.

## Running Mixed Function Experiments

To run experiments with diverse mixed function classes:

1. Use the mixed function configuration to train a model that learns from multiple function classes:

   ```
   python src/train.py --config src/conf/mixed_function.yaml
   ```

2. Customize the mixed function classes in `src/conf/mixed_function.yaml`:

   ```yaml
   # Modify function classes and their distribution
   training:
     task_kwargs:
       function_types: ["linear", "quadratic", "neural_net", "decision_tree"]
       weights: [0.25, 0.25, 0.25, 0.25] # Equal distribution
   ```

3. You can adjust the curriculum learning parameters to control complexity:

   ```yaml
   # Modify the training curriculum
   curriculum:
     points:
       start: 11 # Starting number of context points
       end: 41 # Maximum number of context points
       inc: 2 # Increment size
       interval: 2000 # Steps between increments
   ```

4. For evaluation, the model will automatically test performance on each mixed function class separately, allowing you to analyze how well the model generalizes across different function types.
