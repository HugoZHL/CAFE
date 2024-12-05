# PinSAGE with CAFE

## Requirements

same as [DGL Pinsage example](https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage)

## Prepare datasets

### Nowplaying-rs

1. Download and extract the Nowplaying-rs dataset from https://zenodo.org/record/3248543/files/nowplayingrs.zip?download=1
   into the current directory.
2. Run `python process_nowplaying_rs.py ./nowplaying_rs_dataset ./data_processed`

## Run model

### Nearest-neighbor recommendation

This model returns items that are K nearest neighbors of the latest item the user has
interacted.  The distance between two items are measured by Euclidean distance of
item embeddings, which are learned as outputs of PinSAGE.

- The model can be trained using the following script

   ```bash
   bash run.sh
   ```
