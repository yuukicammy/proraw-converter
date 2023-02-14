# ProRaw Converter

This program converts ProRaw files to sRGB for the purpose of understanding the imaging pipeline of Apple ProRaw.

## Requirements
- C++ compiler supporting C++17 (gcc or Clang)
- [Xtensor](https://github.com/xtensor-stack/xtensor)
- [Xtensor Blas](https://github.com/xtensor-stack/xtensor-blas)
- [Boost](https://github.com/boostorg/boost)
- [OpenCV](https://github.com/opencv/opencv)

## Installation
```bash
$ cd ProRawConversion
$ mkdir build
$ cd build
$ cmake ..
$ make -j
```

## Usage
### Running experiment code
```bash
$ ./experiments/my_conversion -h
The program 1) converts a ProRaw image in sRGB' color space, 2) [optional] adjusts the brightness and contrasts, 3) applys gamma correction, and then 4) saves the result in PNG format. 
If you do not want to adjust brightness and contrast, do not specify the -t option or specify -t 0.
Usage:
  ProRaw Converter [OPTION...] ProRawFilePath

  -r, --raw         Save the raw image
  -d, --debug       Enable debugging. Log file is output to ../logs/.
  -t, --thresh arg  Threshold of histogram stretching in the range [0, 1] 
                    (-t 0.1 recommended). 
                    If this option is not specified, the brightness and 
                    contrast will not be adjusted. -t 0 means no brightness 
                    and contrast adjustment, -t 1 means converting to a 
                    completely black image. (default: 0.)
  -m, --measure     Measure execution speed
  -h, --help        Print usage
```

```bash
$ ./experiments/my_conversion -m -t 0.1 ../data/IMG_0008.DNG
Done conversion from camera native color space to sRGB'. 
 -- Run time (ms): 359.000000
Done adjusting the brightness and contrast.
 -- Run time (ms): 206.000000
Done gamma correction.
 -- Run time (ms): 199.000000
Done all conversion.
 -- Total run time (ms): 764.000000
```

## Results
|No Adjustment|Threshold=0.04|Threshold=0.08|Threshold=0.1|Threshold=0.12|
|---|---|---|---|---|
|![Result image (threshold=0)](data/IMG_0008.DNG.cv_srgb_no_adj.png)|![Result image (threshold=0.04)](data/IMG_0008.DNG.cv_srgb_adj_0.040000.png)|![Result image (threshold=0.08)](data/IMG_0008.DNG.cv_srgb_adj_0.080000.png)|![Result image (threshold=0.10)](data/IMG_0008.DNG.cv_srgb_adj_0.100000.png)|![Result image (threshold=0.12)](data/IMG_0008.DNG.cv_srgb_adj_0.120000.png)|

## License

[MIT](https://choosealicense.com/licenses/mit/)