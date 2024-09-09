# CAFE Artifact Evaluation

This is the evaluation artifact for the SIGMOD 2024 paper: *CAFE: Towards Compact, Adaptive, and Fast Embedding for Large-scale Recommendation Models* ([**PDF**](https://dl.acm.org/doi/10.1145/3639306)), including codes and scripts for reproducing experiments in the paper.
We aim to verify the paper’s **availability** and **reproducibility** through this artifact.

## Environment
### Hardware
Our experiments were conducted in the following hardware environment:
- GPU: 8 NVIDIA RTX TITAN 24 GB.
- CPU: 2 Intel® Xeon® Gold 5120 CPUs @ 2.20GHz.
- Memory: 377GB.
- Disk space: 7.2TB.

For portability, we have the following hardware requirement:
- GPU: 1 card per experiment, with over 16GB memory due to the embedding table size.
- CPU: no requirement, since our tasks are not CPU-intensive.
- Memory: nearly no requirement for training (e.g. 8GB should suffice), since we're using `memmap` for dataloading; however, for data processing, we need 31GB for Avazu, 56 GB for KDD12, 69GB for Criteo, and around 200-300GB for CriteoTB.
- Disk space: 60GB for the 3 smaller dataset, around 1TB for the CriteoTB dataset.

### Software
Our experiments were conducted in the following software environment:
- System: Linux Ubuntu 16.04.
- Python packages: use conda environment, see `environment.yml`.
- C++: use g++ version 9.4.0, openmp version 4.5, and C++17 for compilation.

For portability, we have the following suggestions:
- System: any system that supports conda and g++ is acceptable, including Linux, macOS, and Windows. However, we suggest using Linux, since other systems require some modifications to the conda environment.
- Python packages: please first download and install Anaconda or Miniconda from the website https://www.anaconda.com/, then use the command `conda create --name cafe --file environment.yml` to create a new conda environment for experiments.
- C++: please ensure the following command is runnable: `g++ -fPIC -shared -o tricks/sklib.so --std=c++17 -O3 -fopenmp tricks/sketch.cpp`.


## Code Structure

We present the hierarchical code structure of this artifact as follows:
```
- datasets: process and store the 4 public datasets
    - process_data.py: script to convert raw data into binary data

- excels: store the experimental results

- experiments: run the experiments and save into `excels`
    - run_dataset: main scripts for metrics v.s. CRs and iterations
        - loss_auc_cr_<dataset>.py: scripts for the 4 datasets

    - read_tensorboard: transfer tensorboard file to csv format
        - <dataset>_to_csv.py: parse the results of the 4 datasets

    - sensitivity: script for configuration sensitivity on Criteo
        - hotrate.py & hotrate_tocsv.py: Fig. 15(a) in our paper
        - threshold.py & threshold_tocsv.py: Fig. 15(b) in our paper
        - decay.py & decay_tocsv.py: Fig. 15(c) in our paper

    - sketch: test the performance of HotSketch
        - sketchtest.cpp: HotSketch implementation
        - sketchtest.py: test the performance of HotSketch

    - throughput.py: test the throughput and latency

- pngs: store the experimental pngs

- visualization: visualize results, generating pngs from csvs
    - plot_metric_cr.py: plot AUC/loss v.s. compression ratios
    - plot_metric_cr_mde.py: the same as above, specifically for MDE
    - plot_metric_iter.py: plot AUC/loss v.s. iterations
    - plot_metric_iter_ada.py: the same as above, specifically for AdaEmbed
    - plot_hyper.py: plot configuration v.s. AUC
    - plot_latency.py: plot different methods' latency
    - plot_throughput.py: plot different methods' throughput
    - plot_sketch.py: plot experimental results of HotSketch
```


## Prepare Datasets
We experimented on 4 public datasets: Criteo (or Criteo-Kaggle), CriteoTB (or Criteo-Terabyte), Avazu, KDD12. While Criteo, Avazu, and KDD12 are relatively smaller, CriteoTB is the largest open-source CTR dataset to our knowledge.

In the script below, we download and process these 4 dataset. Please review the comments before running the script, as the datasets need to be downloaded manually.

```bash
mkdir -p datasets; cd datasets

# Criteo-Kaggle
# Please visit https://ailab.criteo.com/ressources/ and click "Kaggle Display Advertising dataset" to download the Criteo dataset (4.3GB) to current directory.
# Here we assume the file `kaggle-display-advertising-challenge-dataset.tar.gz` is already downloaded.
mkdir -p criteo
tar -xvzf kaggle-display-advertising-challenge-dataset.tar.gz -C criteo
python process_data.py --dataset criteo

# Avazu
# Please visit https://www.kaggle.com/c/avazu-ctr-prediction/data, locate and click `train.gz` (1.12GB) in the "Data Explorer", then click the download icon to save the file to current directory. Alternatively, if you have the kaggle API, you can also follow the instructions on this website to download the data.
# Here we assume the file `train.gz` is already downloaded.
mkdir -p avazu
mv train.gz avazu/
gzip -d avazu/train.gz
python process_data.py --dataset avazu

# KDD12
# Please visit https://www.kaggle.com/c/kddcup2012-track2/data, locate and click `track2.zip` (2.88GB) in the "Data Explorer", then click the download icon to save the file to current directory. Alternatively, if you have the kaggle API, you can also follow the instructions on this website to download the data.
# Here we assume the file `track2.zip` is already downloaded.
unzip track2.zip -d kdd12
python process_data.py --dataset kdd12

# Criteo-Terabyte (very TIME-CONSUMING!)
# Please visit https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset and follow the instructions to download the CriteoTB dataset (342GB) to current directory. Due to the large size of the dataset, the downloading (several days) and training process (each around 1 day) can be very time-consuming. We recommend trying other datasets first and only using this one if you have sufficient time and resources.
mkdir -p criteotb
# download 24 days data into criteotb
gzip -d criteotb/day_*.gz
python process_data.py --dataset criteotb

cd ..
```

## Experiments

In the [Code Structure](#code-structure) section, we have introduce the experiment scripts for experiments and visualization. In this section, we will walk through the experiment presented in our paper using these scripts.

### Metrics v.s. Compression Ratios
Most of the experiments focus on metrics (AUC and loss) versus compression ratios. In our paper, Fig. 8, 10(a-b), 11, 12, 14(a), 16, and 17(a-b) fall into this category. Please replace `<dataset>` with the 4 datasets, and run the scripts:

```bash
mkdir -p excels
cd experiments/run_dataset
python loss_auc_cr_<dataset>.py
cd ../read_tensorboard
python <dataset>_to_csv.py
cd ../..
```

And the results will be shown in `excels/<dataset>/auc.csv` and `excels/<dataset>/loss.csv`.

After obtaining the AUC and loss for all methods at different compression ratios, we can plot the AUC/loss v.s. compression ratios using the following scripts:

```bash
python visualization/plot_metric_cr.py
python visualization/plot_metric_cr_mde.py
```

The scripts generate the experimental figures for Fig. 8, 10(a-b), and 12.

### Metrics v.s. Iterations
We also check the metrics versus iterations. In our paper, Fig. 9, 10(c), 14(b-c), and 17(c) fall into this category.
The scripts of the previous section already generate the experimental results in `excels/<dataset>/<metric><compress ratio>_<type>.csv`. We use the following scripts to plot figures.

```bash
python visualization/plot_metric_iter.py
python visualization/plot_metric_iter_ada.py
```

The scripts generate the experimental figures for Fig. 9 and 10(c).

### Throughput & Latency
Fig. 13 in our paper exhibits the throughput and latency of different methods.

```bash
cd experiments
python throughput.py
cd ..
```

The results are saved in the `excels/throughput.csv` file, and we can plot the figures using:

```bash
python visualization/plot_throughput.py
python visualization/plot_latency.py
```

### Sensitivity
We evaluate model performance across various configurations, and the results are shown in Fig. 15 in our paper.

```bash
cd experiments/sensitivity

# Fig. 15(a), different hot percentage. The result is in excels/hotrate.auc.
python hotrate.py
python hotrate_tocsv.py

# Fig. 15(b), different threshold of HotSketch. The result is in excels/threshold.auc.
python threshold.py
python threshold_tocsv.py

# Fig. 15(c), different decay coefficient of HotSketch. The result is in excels/decay.auc.
python decay.py
python decay_tocsv.py

cd ../..
```

And please use the following script for visualization:
```bash
python visualization/plot_hyper.py
```

### HotSketch
Fig. 18 in our paper presents the recall and throughput of the HotSketch.

```bash
cd experiments/sketch
python sketchtest.py
cd ../..
```

The results are saved in `excels/sketch/`, and the following script is for visualization:
```bash
python visualization/plot_sketch.py
```

## Discussion
For customization, users can utilize our code to explore the supported flags and add new methods to evaluate performance on these 4 datasets.

For portability, our code has minimial dependencies on specific hardware or software. Please refer to the [Environment](#environment) section for details.

For replicability, although we set the random seed in our code, it’s worth noting that the actual computing kernels on different hardware environments (e.g., different GPU cards) may vary, leading to different results. However, the model’s AUC/loss and efficiency should not differ significantly from our experiments.

Due to the time-consuming nature of our experiments, we may not be able to fully validate all our experiment scripts. If you encounter any issues while running our code, please feel free to raise them on our github repo.
