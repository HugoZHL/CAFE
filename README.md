### CAFE: Towards Compact, Adaptive, and Fast Embedding for Large-scale Recommendation Models

Our implementation builds upon DLRM repo: https://github.com/facebookresearch/dlrm

1. The code supports interface with the [Criteo Kaggle Display Advertising Challenge Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).

   - The model can be trained using the following script

     - Convert the value of the numerical feature to log(x)+1.
     - Ensure that the feature count for each field is independent.
     - Set the parameters cat_path, dense_path, label_path and count_path in the script.

     ```
     ./bench/criteo_kaggle.sh
     ```

2. The code supports interface with the [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).

   - Please do the following to prepare the dataset for use with this code:

     - Convert the value of the numerical feature to log(x)+1.
     - Ensure that the feature count for each field is independent.
     - Set the parameters cat_path, dense_path, label_path and count_path in the script.

   - The model can be trained using the following script

     ```
     ./bench/criteo_terabyte.sh
     ```

3. The code also supports another two datasets [Avazu](https://kaggle.com/competitions/avazu-ctr-prediction) and [KDD12](https://kaggle.com/competitions/kddcup2012-track2).
   - Please do the following to prepare the dataset for use with this code:
     - Ensure that the feature count for each field is independent.
     - Set the parameters cat_path, dense_path, label_path and count_path in the script.

   - The model can be trained using the following script

     ```
     ./bench/avazu.sh
     ./bench/kdd12.sh
     ```

4. The code provides three models to train the dataset:
   - dlrm:

     ```
     ./bench/criteo_terabyte.sh
     ```
   - wdl:

     ```
     ./bench/wdl.sh
     ```
   - dcn:

     ```
     ./bench/dcn.sh
     ```

5. The code provides six methods for generating embedding layers:

   - Full embedding with the following script

     ```
     ./bench/criteo_terabyte.sh
     ```

   - Hash embedding with the following script

     ```
     ./bench/criteo_terabyte.sh "--hash-flag --compress-rate=0.001"
     ```

   - CAFE with the following script

     ```
     ./bench/criteo_terabyte.sh "--sketch-flag --compress-rate=0.001 --hash-rate=0.3"
     ```

   - QR embedding with the following script

     ```
     ./bench/criteo_terabyte.sh "--qr-flag --qr-collisions=10"
     ```

   - Ada embedding with the following script

     ```
     ./bench/criteo_terabyte.sh "--ada-flag --compress-rate=0.1"
     ```
   - MD embedding with the following script

     ```
     ./bench/criteo_terabyte.sh "--md-flag --compress-rate=0.1"
     ```