# super-resolution
A collection of super-resolution models & algorithms

Detail introduction of each model is in corresponding sub-folds.

Authored by [icpm](https://github.com/icpm)

## Requirement
- python3.6
- numpy
- pytorch 1.0.0

## Models
- [VDSR](https://github.com/icpm/super-resolution/tree/master/VDSR)
- [EDSR](https://github.com/icpm/super-resolution/tree/master/EDSR)
- [DCRN](https://github.com/icpm/super-resolution/tree/master/DRCN)
- [SubPixelCNN](https://github.com/icpm/super-resolution/tree/master/SubPixelCNN)
- [SRCNN](https://github.com/icpm/super-resolution/tree/master/SRCNN)
- [FSRCNN](https://github.com/icpm/super-resolution/tree/master/FSRCNN)
- [SRGAN](https://github.com/icpm/super-resolution/tree/master/SRGAN)
- [DBPN](https://github.com/icpm/super-resolution/tree/master/DBPN)

## Usage
train:

```bash
$ python3 main.py -m [sub/srcnn/cdsr/edsr/fsrcnn/drcn/srgan/dbpn]
```

super resolve:

```bash
$ python3 super_resolve
```