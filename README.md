# Foreground background




### Environment setup

Start by creating new Anaconda environment using definition in `conda.yml`.
```
conda env create -f conda.yml

```

### Train models

1. Make sure to update `data_dir` entry in the corresponding config file in the `./config` directory.
2. Run the training by specifying the desired experiment configuration, e.g.
```
conda activate frg-bgr-modelling

python train.py -c eurasianlynx_PITS.json
```
3. The model is by default created in `./outputs` directory

### Test models

Run the `test.py` script, using path of the model output from the previous step, e.g.

```
python test.py -o outputs\models\EurasianLynx_PITS\20240820_171509 

Accuracy without prior = 52.52%
Accuracy without prior in new locations only = 19.14%
Expected Calibration Error (ECE) without prior = 0.312 
Accuracy with prior MovingLocationPrior = 60.98%
Accuracy with prior MovingLocationPrior in new locations only = 19.62%
Expected Calibration Error (ECE) with prior MovingLocationPrior = 0.276
```
