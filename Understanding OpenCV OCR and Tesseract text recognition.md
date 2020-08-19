## Understanding OpenCV OCR and Tesseract text recognition

[![img](https://pyimagesearch.com/wp-content/uploads/2018/09/opencv_ocr_pipeline.png)](https://pyimagesearch.com/wp-content/uploads/2018/09/opencv_ocr_pipeline.png)

Now that we have OpenCV and Tesseract successfully installed on our system we need to briefly review our pipeline and the associated commands.

To start, we’ll apply [OpenCV’s EAST text detector](https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/) to detect the presence of text in an image. The EAST text detector will give us the bounding box *(x, y)*-coordinates of text ROIs.

We’ll extract each of these ROIs and then pass them into Tesseract v4’s LSTM deep learning text recognition algorithm.

The output of the LSTM will give us our actual OCR results.

Finally, we’ll draw the OpenCV OCR results on our output image.

But before we actually get to our project, let’s briefly review the Tesseract command (which will be called under the hood by the pytesseract library). When calling the tessarct binary we need to supply a number of flags. The three most important ones are -l, --oem, and  --psm.

The -l flag controls the language of the input text. We’ll be using eng (English) for this example but you can see all the languages Tesseract supports [here](https://github.com/tesseract-ocr/tesseract/wiki/Data-Files).

The --oem argument, or OCR Engine Mode, controls the type of algorithm used by Tesseract.

You can see the available OCR Engine Modes by executing the following command:

```bash
$ tesseract --help-oem
OCR Engine modes:
  0    Legacy engine only.
  1    Neural nets LSTM engine only.
  2    Legacy + LSTM engines.
  3    Default, based on what is available.
```
We’ll be using --oem 1 to indicate that we wish to use the deep learning LSTM engine only.

The final important flag, --psm controls the automatic Page Segmentation Mode used by Tesseract:

``` bash
$ tesseract --help-psm
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR.
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.
```

For OCR’ing text ROIs I’ve found that modes 6 and 7 work well, but if you’re OCR’ing large blocks of text then you may want to try 3 , the default mode.

Whenever you find yourself obtaining incorrect OCR results I highly recommend adjusting the --psm as it can have dramatic influences on your output OCR results.

## Implementing our OpenCV OCR algorithm

We are now ready to perform text recognition with OpenCV! Open up the text_recognition.py file and insert the following code:

``` python
# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
```

Today’s OCR script requires five imports, one of which is built into OpenCV. Most notably, we’ll be using pytesseract and OpenCV. My imutils package will be used for [non-maxima suppression](https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/) as OpenCV’s NMSBoxes function doesn’t seem to be working with the Python API. I’ll also note that NumPy is a dependency for OpenCV.

The argparse package is included with Python and handles command line arguments — there is nothing to install.