# NMR-TS
Molecule identification from NMR spectrum using de novo molecule generator 

This code is for reference only as it requires MPI and Gaussian environment which is hard to ensure among different clusters. Also, for the Gaussian part, we have used a Gaussain encapsulation package which we can not make public for now. Replacing this part by a traditional Gaussian code or other NMR spectrum prediction method would solve the problem.  

## Requirements

1.Python

2.RDKit

3.TensorFlow

4.Gaussian

## How to use?

Clone repo
```
git clone https://github.com/tsudalab/NMR-TS.git
cd NMR-TS
```

Train the RNN model*

```
cd train_RNN
python train_RNN.py
```
*You need a SMILES dataset for this step

Run main program
```
python mpi_thread_ChemTS_tree_vl_jz.py
```
# License
This project is licensed under the terms of the MIT license.
