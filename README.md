This repository is a copy of [OpenSkyTraffic](https://github.com/Weijiang-Xiong/OpenSkyTraffic).
Please take a look at their repository for more information about SkyTraffic. 

It will be used to study different temporal encoders (LSTM, bidirectional LSMT, GRU) and pooling methods. 

To run first clone and install dependencies:
```
git clone https://github.com/Weijiang-Xiong/OpenSkyTraffic.git
python -m pip install -e .
```

Download dataset:
```
python -m pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Greatriver/SimBarca', repo_type='dataset', local_dir='datasets/simbarca')"
```

Training:
```
python scripts/train.py --config-file PATH/TO/CFG/FILE.py train.output_dir=scratch/SAVE_DIR
```

Visualizing results:
```
python scripts/train.py --eval-only --config-file scratch/SAVE_DIR/config.yaml evaluation.visualize=True
```
