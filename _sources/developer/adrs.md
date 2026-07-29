# Architecture Decision Records

## ADR-001: Allow up to 2 dimensional coordinates/mask as coordinates and masks.

At the beginning we only allowed scalar or 1 dimensional coordinates or masks in the metadata.
It is because the metadata is stored as plain text in the tiff file.
If we allow arbitrary dimensional coordinates or masks, the metadata size may exceed the size of the image file itself.
There is no strict rule about the size of the metadata in the tiff format itself.
We try to keep the metadata size small as possible to increase the usability and reduce potential tourbles such as storage size or loading latency.

However, there was need for 2 dimensional coordinates, especially for the coordinate that depends on the pixel position.
For example, when a tiff file is a histogram of (x, y, tof), in order to compute wavelength, we need `Ltotal` of each pixel, which is a 2 dimensional (x, y) coordinate.
This usecase was found in the detector position calibration routine.

