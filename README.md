# Run training VAE
```sh
python train.py --model=VAE --lr-scheduler=plateau --max-epochs=70 --batch-size=28
```

# Run training Difussion
```sh
python train.py --model=VAE --lr-scheduler=plateau --max-epochs=70 --batch-size=28
```


# Sample from VAE
```sh
python sample.py --model=VAE --number-samples=10 
```