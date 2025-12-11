# THR
Source code of THR, which is proposed in this paper:
Untangling Structural Bottlenecks: Reassessing Heterophilic Graphs via Torque Learning
### Requirement
- Python == 3.9.12
- PyTorch == 1.11.0
- Numpy == 1.21.5
- Scikit-learn == 1.1.0
- Scipy == 1.8.0
- Texttable == 1.6.4
- Tensorly == 0.7.0
- Tqdm == 4.64.0

### Quick Running
For example, to reproduce GCN with THR on the Texas dataset, run the following

`python train.py --dataset texas --net GCN`
