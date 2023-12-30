Python Implementation for Harris Corner Detector
===================

## Requirements
- NumPy
- Matplotlib
- Pillow

## Sources
- Image from Davidwkennedy - http://en.wikipedia.org/wiki/File:Bikesgray.jpg - [License](https://creativecommons.org/licenses/by-sa/3.0/deed.en)
- [Harris Corner Detector Wikipedia](https://en.wikipedia.org/wiki/Harris_corner_detector)

## Algorithm
- Compute spatial derivative in x and y direction with Sobel kernel
- Convolve with gaussian filter
- Setup Tensor M
- Compute det and trace
- Analyse Harris Response
- Non-maximum suppression