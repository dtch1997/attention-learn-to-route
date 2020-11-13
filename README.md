# An Attention-Based, Reinforcement-Learned Heuristic Solver for the Double Travelling Salesman Problem with Multiple Stacks

Attention based model for learning to solve the Double Travelling Salesman Problem with Multiple Stacks (DTSPMS). Trained with REINFORCE with greedy rollout baseline. This work extends an approach first used in [Kool, 2019](https://github.com/wouterkool/attention-learn-to-route) for the TSP and other simple variants to the DTSPMS, a relatively more difficult routing problem. 

## Dependencies

* Python>=3.6
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)

## Quick start

For training DTSPMS instances with 20 nodes and using rollout as REINFORCE baseline:
```bash
python run.py --problem dtspms --graph_size 20 --baseline rollout --run_name 'dtspms20_rollout'
```

## Usage

### Generating data

Training data is generated on the fly. To generate test data for the DTSPMS:
```bash
python generate_data.py --problem dtspms --name test --seed 4321
```

Benchmark instances from TSPLIB are provided as well. To parse those into a format readable by the model:
```bash
python generate_data.py --problem dtspms --from_benchmark all --name filler
```


### Training

For training DTSPMS instances with 20 nodes and using rollout as REINFORCE baseline:
```bash
python run.py --problem dtspms --graph_size 20 --baseline rollout --run_name 'dtspms20_rollout'
```

#### Multiple GPUs
By default, training will happen *on all available GPUs*. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=2,3 python run.py 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 nodes).

#### Warm start
You can initialize a run using a pretrained model by using the `--load_path` option:
```bash
python run.py --problem dtspms --graph_size 100 --load_path pretrained/dtspms_20/epoch-31.pt
```

The `--load_path` option can also be used to load an earlier run, in which case also the optimizer state will be loaded:
```bash
python run.py --problem dtspms --graph_size 20 --load_path 'outputs/dtspms_20/dtspms20_rollout_{datetime}/epoch-0.pt'
```

The `--resume` option can be used instead of the `--load_path` option, which will try to resume the run, e.g. load additionally the baseline state, set the current epoch/step counter and set the random number generator state.

### Evaluation
To evaluate a model, you can add the `--eval-only` flag to `run.py`, or use `eval.py`, which will additionally measure timing and save the results. 
```bash
python eval.py data/dtspms/dtspms20_test_seed4321.pkl --model pretrained/dtspms_20 --decode_strategy greedy
```
If the epoch is not specified, by default the last one in the folder will be used.

### Other options and help
```bash
python run.py -h
python eval.py -h
```

## Acknowledgements
This approach is not original, but first presented in [Kool, 2019](https://github.com/wouterkool/attention-learn-to-route). Thanks to the AA228 teaching staff for advice and technical discussion, as well as Google for funding this project with cloud credits. 
