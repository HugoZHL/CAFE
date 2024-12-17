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
- Memory: nearly no requirement for training (e.g. 8GB should suffice), since we're using `memmap` for dataloading; however, for data processing, we need 31GB for Avazu, 56 GB for KDD12, 69GB for Criteo, and several hundreds GB for CriteoTB (if CPU OOM issue occurs and interrupts the processing, please re-init the data processing script and skip the processed days manually).
- Disk space: 60GB for the 3 smaller dataset, around 1TB for the CriteoTB dataset.

### Software
Our experiments were conducted in the following software environment:
- System: Linux Ubuntu 16.04.
- Python packages: use conda environment, see `environment.yml`.
- C++: use g++ version 9.4.0, openmp version 4.5, and C++17 for compilation.
- CUDA: 11.2 .

To configure the environment, please follow these instructions:
- System: any system that supports conda and g++ is acceptable, including Linux, macOS, and Windows. However, we suggest using Linux, since other systems require some modifications to the conda environment.
- Python packages: please first download and install Anaconda or Miniconda from the website https://www.anaconda.com/, then use the following command to create a new conda environment named `cafe` for experiments.
```bash
conda env create -f environment.yml
```
- C++: please use the following command to compile C++ codes for CAFE: 
```bash
g++ -fPIC -shared -o embeddings/sklib.so --std=c++17 -O3 -fopenmp embeddings/sketch.cpp
```


## Code Structure

We present the hierarchical code structure of this artifact as follows:
```
- board: store the experimental results (tensorboard and other information)

- datasets: process and store the datasets (4 datasets + 1 variant)
    - process_data.py: script to convert raw data into binary data

- embeddings: different embedding compression methods
    - ada_embedding_bag.py: AdaEmbed
    - hash_embedding_bag.py: Hash
    - init_embed.py: initialize embedding layers
    - md_embedding_bag.py: MDE
    - qr_embedding_bag.py: Q-R trick
    - sk_embedding_bag.py: CAFE
    - sketch.cpp: CAFE's C++ codes

- pngs: store the experimental PNGs

- sketch_expr: scripts and results for sketch experiments
    - sketchtest.cpp: C++ codes for sketch experiments
    - sketchtest.py: sketch experiments scripts

- tasks: json configuration files for experiment tasks
    - sensitivity: configuration files of sensitivity experiments
        - decay.json: cafe importance decay ratio
        - hot_percentage.json: percentage of hot embeddings
        - separate_field.json: separate HotSketch for each field
        - threshold.json: hot/cold threshold
        - use_freq.json: use frequency instead of gradient norm
    - avazu.json: configurations for DLRM on Avazu dataset
    - criteo.json: configurations for DLRM on Criteo dataset
    - criteotb.json: configurations for DLRM on CriteoTB dataset
    - dcn_criteo.json: configurations for DCN on CriteoTB dataset
    - kdd12.json: configurations for DLRM on KDD12 dataset
    - latency.json: configurations for latency tests
    - wdl_criteo.json: configurations for WDL on CriteoTB dataset

- visualization: visualize results, generating pngs from csvs
    - board_reader.py: read tensorboard results
    - plot_hyper.py: plot configurations v.s. AUC
    - plot_latency.py: plot different methods' latency and throughput
    - plot_metric_cr.py: plot AUC/loss v.s. compression ratios
    - plot_metric_iter.py: plot AUC/loss v.s. iterations
    - plot_sketch.py: plot experimental results of HotSketch

- job_scheduler.py: schedule tasks on GPUs; 
    usage: python job_scheduler.py tasks/xxx.json tasks/yyy.json 

- load_data.py: initialize dataset and dataloader for training/testing

- main.py: train/test different methods

- models.py: model structures of DLRM, DCN, WDL
```


## Prepare Datasets
We experimented on 4 public datasets: Criteo (or Criteo-Kaggle), CriteoTB (or Criteo-Terabyte), Avazu, KDD12. Additionally, we have a variant dataset CriteoTB-1/3. While Criteo, Avazu, and KDD12 are relatively smaller, CriteoTB is the largest open-source CTR dataset to our knowledge.

In the script below, we download and process these datasets. Please review the comments before running the script, as the datasets need to be downloaded manually.

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

# Criteo-Terabyte (very TIME-CONSUMING! We don't recommend reproducing this dataset.)
# Please visit https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset and follow the instructions to download the CriteoTB dataset (342GB) to current directory. Due to the large size of the dataset, the downloading (several days), processing (several days, with manually checking memory usage), and training process (each around 1 day) can be very time-consuming. We recommend trying other datasets first and only using this one if you REALLY have sufficient time and resources.
mkdir -p criteotb
# download 24 days data into criteotb
gzip -d criteotb/day_*.gz
python process_data.py --dataset criteotb

# CriteoTB-1/3
# A variant of CriteoTB, use only the 1, 4, 7, ..., 22 days for training and the 24th day for testing. The processing is based on CriteoTB, so please process CriteoTB first.
mkdir -p criteotb13
python process_data.py --dataset criteotb13

cd ..
```

## Experiments

In the [Code Structure](#code-structure) section, we have introduce the experiment scripts for experiments and visualization. In this section, we will walk through the experiment presented in our paper using these scripts.

### Metrics v.s. Compression Ratios / Iterations
Most of the experiments focus on metrics (AUC and loss) versus compression ratios or iterations. In our paper, Fig. 8, 9, 10, 11, 12, 14, 16, and 17 fall into this category. Please check the following scripts:

```bash
# first run all the tasks
# run Criteo
python job_scheduler.py tasks/criteo.json
# run CriteoTB; only run this script if you have processed CriteoTB dataset!
python job_scheduler.py tasks/criteotb.json
# run Avazu
python job_scheduler.py tasks/avazu.json
# run KDD12
python job_scheduler.py tasks/kdd12.json
# run CriteoTB-1/3; ; only run this script if you have processed CriteoTB(-1/3) dataset!
python job_scheduler.py tasks/criteotb13.json

# run WDL on CriteoTB; only run this script if you have processed CriteoTB dataset!
python job_scheduler.py tasks/wdl_criteotb.json
# run DCN on CriteoTB; only run this script if you have processed CriteoTB dataset!
python job_scheduler.py tasks/dcn_criteotb.json

# then plot the figures
# plot metrics v.s. compression ratios
# please check the following script first
#  and EXCLUDE those tasks you did not run!!!
python visualization/plot_metric_cr.py
# plot metrics v.s. iterations
# please check the following script first
#  and EXCLUDE those tasks you did not run!!!
python visualization/plot_metric_iter.py
```

Then we have the following figures:
| Path to figure | Figure in paper |
| --- | --- |
| pngs/criteo_auc_cr_fhqac.png | Fig. 8(a) |
| pngs/criteotb_auc_cr_fhqac.png | Fig. 8(b) |
| pngs/criteo_loss_cr_fhqac.png | Fig. 8(c) |
| pngs/criteotb_loss_cr_fhqac.png | Fig. 8(d) |
| pngs/criteo0.01_auc_iter_fhqc.png | Fig. 9(a) |
| pngs/criteotb0.01_auc_iter_hqc.png | Fig. 9(b) |
| pngs/criteo0.2_auc_iter_fac.png | Fig. 9(c) |
| pngs/criteotb0.02_auc_iter_ac.png | Fig. 9(d) |
| pngs/criteo0.01_loss_iter_fhqc.png | Fig. 9(e) |
| pngs/criteotb0.01_loss_iter_hqc.png | Fig. 9(f) |
| pngs/criteo0.2_loss_iter_fac.png | Fig. 9(g) |
| pngs/criteotb0.02_loss_iter_ac.png | Fig. 9(h) |
| pngs/kdd12_auc_cr_fhqac.png | Fig. 10(a) |
| pngs/avazu_loss_cr_fhqac.png | Fig. 10(b) |
| pngs/avazu0.2_loss_iter_fhqac.png | Fig. 10(c) |
| pngs/wdl_criteotb_auc_cr_hqac.png | Fig. 11(a) |
| pngs/wdl_criteotb_loss_cr_hqac.png | Fig. 11(b) |
| pngs/dcn_criteotb_auc_cr_hqac.png | Fig. 11(c) |
| pngs/dcn_criteotb_loss_cr_hqac.png | Fig. 11(d) |
| pngs/criteo_auc_cr_fhmc.png | Fig. 12(a) |
| pngs/criteotb_auc_cr_fhmc.png | Fig. 12(b) |
| pngs/criteo_loss_cr_fhmc.png | Fig. 12(c) |
| pngs/criteotb_loss_cr_fhmc.png | Fig. 12(d) |
| pngs/criteo_auc_cr_foc.png | Fig. 14(a) |
| pngs/criteo0.001_auc_iter_foc.png | Fig. 14(b) |
| pngs/criteo0.001_loss_iter_foc.png | Fig. 14(c) |
| pngs/criteotb13_auc_cr_hqac.png | Fig. 17(a) |
| pngs/criteotb13_loss_cr_hqac.png | Fig. 17(b) |
| pngs/criteotb130.02_loss_iter_hqac.png | Fig. 17(c) |



### Throughput & Latency
Fig. 13 in our paper exhibits the throughput and latency of different methods. When running these tasks, please ensure that the GPUs are not used by other programs.

```bash
# run tasks
python job_scheduler.py tasks/latency.json

# plot figures
python visualization/plot_latency.py
```

Then we have the following figures:
| Path to figure | Figure in paper |
| --- | --- |
| pngs/latency.png | Fig. 13(a) |
| pngs/throughput.png | Fig. 13(b) |

### Sensitivity
We evaluate model performance across various configurations, and the results are shown in Fig. 15 in our paper.

```bash
# run tasks
python job_scheduler.py \
    tasks/sensitivity/decay.json \
    tasks/sensitivity/hot_percentage.json \
    tasks/sensitivity/separate_field.json \
    tasks/sensitivity/threshold.json \
    tasks/sensitivity/use_freq.json

# plot figures
python visualization/plot_hyper.py
```

Then we have the following figures:
| Path to figure | Figure in paper |
| --- | --- |
| pngs/toppercent.png | Fig. 15(a) |
| pngs/threshold.png | Fig. 15(b) |
| pngs/decay.png | Fig. 15(c) |
| pngs/others.png | Fig. 15(d) |

### HotSketch
Fig. 18 in our paper presents the recall and throughput of the HotSketch.

```bash
# run experiments
cd sketch_expr
python sketchtest.py
cd ..

# plot figures
python visualization/plot_sketch.py
```

Then we have the following figures:
| Path to figure | Figure in paper |
| --- | --- |
| pngs/sketch_mem_recall.png | Fig. 18(a) |
| pngs/sketch_throughput.png | Fig. 18(b) |
| pngs/time_recall_100.png | Fig. 18(c) |
| pngs/time_recall_1000.png | Fig. 18(d) |


## Discussion
For customization, users can utilize our code to explore the supported flags and add new methods to evaluate performance on these 4 datasets.

For portability, our code has minimial dependencies on specific hardware or software. Please refer to the [Environment](#environment) section for details.

For replicability, although we set the random seed in our code, it’s worth noting that the actual computing kernels on different hardware environments (e.g., different GPU cards) may vary, leading to different results. However, the model’s AUC/loss and efficiency should not differ significantly from our experiments.

Due to the time-consuming nature of our experiments, we may not be able to fully validate all our experiment scripts. If you encounter any issues while running our code, please feel free to raise them on our github repo.
