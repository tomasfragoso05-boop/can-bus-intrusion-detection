# Data

The dataset is not included in this repository (about 5 GB once processed).

1. Download the **Car-Hacking Dataset** from the HCRL website (Hacking and Countermeasure Research Lab, Korea University):
   https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
2. Put the raw files in `data/raw/`:
   ```
   data/raw/normal_run_data.txt
   data/raw/DoS_dataset.csv
   data/raw/Fuzzy_dataset.csv
   data/raw/gear_dataset.csv
   data/raw/RPM_dataset.csv
   ```
3. Run the preprocessing scripts from the repository root:
   ```
   python preprocessing/data_cleaning.py        # data/raw -> data/processed
   python preprocessing/feature_engineering.py  # data/processed -> data/feature
   ```

To keep the data somewhere else, set the `CAN_DATA_DIR` environment variable to a folder with the same `raw/`, `processed/` and `feature/` layout.

Please cite the dataset authors if you use it:
> E. Seo, H. M. Song, H. K. Kim, "GIDS: GAN based Intrusion Detection System for In-Vehicle Network", 16th Annual Conference on Privacy, Security and Trust (PST), 2018.
