# Run training VAE
```sh
python generative/train.py --model=VAE --lr-scheduler=plateau --max-epochs=70 --batch-size=28
```

# Run training Difussion
```sh
python generative/train.py --model=DIFFUSION --lr-scheduler=plateau --max-epochs=70 --batch-size=28
```


# Sample from VAE
```sh
python generative/sample.py --model=VAE --number-samples=10 
```