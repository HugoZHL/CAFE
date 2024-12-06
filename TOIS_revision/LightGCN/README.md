# CAFE+ LightGCN

This is the implementation of CAFE+ in LightGCN.

This code is based on on LightGCN based on [link](https://github.com/xurong-liang/CERP).

- The model can be trained using the following script after prepare the datasets.

   ```bash
   cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --topks="[20]" --recdim=64
   ```
