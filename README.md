# robbie

![Logo](res/logo_text.png "Seldon Logo")

## About
Library for home-grown neural networks

## Installation 

Assuming you are using some kind of `conda` environment, here is how you can install `robbie` and use `robbielib` in `python`. 

```bash
meson setup build --prefix $CONDA_PREFIX
meson compile -C build
meson install -C build 
```

Now you can run import `robbielib` from anywhere if your conda environment is activated.
