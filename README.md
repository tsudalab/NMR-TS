# NMR-TS
Molecule identification from NMR spectrum using de novo molecule generator 

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
