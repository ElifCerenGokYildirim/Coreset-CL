# Exploring Continual Learning Through Data Diet

<img src="teaser.png" alt="Teaser" width="700"/>

## Introduction

Existing continual learning (CL) methods learn from all the available data. However, this is not the case in human cognition, which efficiently focuses on key experiences while disregarding redundant information. Similarly, not all data points in a dataset have equal potential; some can be more representative or informative than others. This disparity in data points may significantly impact incremental performance, as both the quality and quantity of samples directly influence the model’s effectiveness and efficiency.

Drawing inspiration from this, we explore the potential of learning from important samples, particularly through the application of coreset selection techniques—a relatively unexplored area in the continual learning literature. Through a comprehensive empirical analysis, we evaluate the performance of various CL learners trained with selected samples and investigate the learning-forgetting dynamics, shedding light on the underlying mechanisms driving their improved stability-plasticity balance.

We present several significant observations:
1. Learning from selectively chosen samples enhances incremental accuracy.
2. Retaining knowledge.
3. Improving learned representations of previous tasks.

Our findings contribute to a deeper understanding of selective learning strategies in CL scenarios. We built upon the [PYCIL library](https://github.com/G-U-N/PyCIL) and use [DeepCore](https://github.com/PatrickZH/DeepCore) for the coreset selection methods. We thank these repositories for providing helpful components.

## Dependencies

For the PYCIL:
1. [torch 1.81](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [tqdm](https://github.com/tqdm/tqdm)
4. [numpy](https://github.com/numpy/numpy)
5. [scipy](https://github.com/scipy/scipy)
6. [quadprog](https://github.com/quadprog/quadprog)
7. [POT](https://github.com/PythonOT/POT)

## Datasets

The CIFAR10 and CIFAR100 datasets will be automatically downloaded. To train on another dataset, specify the dataset folder in `utils/data.py`. For further details, please refer to the PYCIL library.

## Coreset Methods

In the `selection` directory, we have implementations of:
- random
- herding
- uncertainty
- forgetting
- submodular (graphcut) methods

## Continual Learners

In the `models` directory, we have implementations of:
- der
- foster
- memo
- icarl
- er
- lwf

## Run Experiment

To run an experiment, edit the `[MODEL NAME].json` file for all settings like `dataset`, `memory_per_class`, `init_cls`, `increment`, `convnet`, `seed`, and `selection_method`. Then, run:

```bash
python main.py --config=./exps/[MODEL NAME].json
