# A Beginner’s Guide to Tesseract OCR


![Image for post](https://miro.medium.com/max/2400/0*4ZLMX7XGQxTGtvUp.jpg)

Image taken from https://en.wikipedia.org/wiki/Optical_character_recognition

This article is a step-by-step tutorial in using Tesseract OCR to recognize characters from images using Python. Due to the nature of Tesseract’s training dataset, digital character recognition is preferred, although Tesseract OCR can also be used for handwriting recognition. Tesseract OCR is an open-source project, started by Hewlett-Packard. Later Google took over development. As of October 29, 2018, the latest stable version 4.0.0 is based on LSTM (long short-term memory). Check it out on [Github](https://github.com/tesseract-ocr/tesseract) to learn more.

The official version of Tesseract OCR allows developers to build their own application using C or C++ API. Over time the community created their own versions of external tools, wrappers, and even training projects. In this article, I will be using a Python wrapper called [tesserocr](https://github.com/sirfz/tesserocr) because:

1. It is simple and easy-to-use
2. It supports version 4.0.0 (at the time of this writing)
3. The comments and explanation in the file are highly detailed

If you are looking for other wrappers or tools, check put this [Github](https://github.com/tesseract-ocr/tesseract/wiki/AddOns#tesseract-wrappers) link.

------

This tutorial consists of the following sections:

1. Setup and installation
2. Preparing test images
3. Usage and API call
4. Fine-tuning
5. Results
6. Conclusion

# 1. Setup and installation

There are multiple ways to install tesserocr. The requirements and steps stated in this section will be based on installation via **pip** on Windows operating system. You can check the steps required via the official [Github](https://github.com/sirfz/tesserocr) if you wanted to install via other methods.

## Github files

Clone or download the [files ](https://github.com/sirfz/tesserocr)to your computer. Once you have completed the download, extract them to a directory. Make sure you have saved it in an easily accessible location — we will be storing the test images in the same directory. You should have a **tesserocr-master** folder that contains all the required files. Feel free to rename it.

## Python

You should have python installed with version 3.6 or 3.7. I will be using Python 3.7.1 installed in a virtual environment for this tutorial.

## Python modules via pip

Download the required [file ](https://github.com/simonflueckiger/tesserocr-windows_build/releases)based on the python version and operating system. I downloaded [tesserocr v2.4.0 — Python 3.7–64bit](https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.4.0-tesseract-4.0.0/tesserocr-2.4.0-cp37-cp37m-win_amd64.whl) and saved it to the **tesserocr-master** folder (you can save it anywhere as you like). From the directory, open a command prompt (simply point it to the directory that holds the *whl* file if you opened a command prompt from other directory). Installation via pip is done via the following code:

```
pip install <package_name>.whl
```

**Package_name** refers to the name of the **whl** file you have downloaded. In my case, I have downloaded *tesserocr-2.4.0-cp37-cp37m-win_amd64.whl*. Hence, I will be using the following code for the installation:

```
pip install tesserocr-2.4.0-cp37-cp37m-win_amd64.whl
```

The next step is to install **Pillow,** a module for image processing in Python. Type the following command:

```
pip install Pillow
```

## Language data files

Language data files are required during the initialization of the API call. There are three types of data files:

1. **tessdata**: The standard model that only works with Tesseract 4.0.0. Contains both legacy engine (--oem 0) and LSTM neural net based engine (--oem 1). **oem** refers to one of the parameters that can be specified during initialization. A lot faster than tessdata_best with with lower accuracy. Link to standard [tessdata](https://github.com/tesseract-ocr/tessdata).
2. **tessdata_best**: Best trained model that only works with Tesseract 4.0.0. It has the highest accuracy but a lot slower compared to the rest. Link to [tessdata_best](https://github.com/tesseract-ocr/tessdata_best).
3. **tessdata_fast**: This model provides an alternate set of integerized LSTM models which have been built with a smaller network. Link to [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast).

I will be using the standard **tessdata** in this tutorial. Download it via the link above and place it in the root directory of your project. In my case, it will be under **tesserocr-master** folder. I took an extra step and renamed the data files as **tessdata**. This means I have the following folder structure:

```
.../tesserocr-master/tessdata
```

# 2. Preparing test images

## Saving images

The most efficient ways to get test images are as follows:

1. Search images online using keywords such as “road sign”, “restaurant” “menus”, “scanned”, “receipts” etc.
2. Use snipping tool to save images of online articles, novels, e-books and etc.
3. Use camera to take screenshot of labels or instructions pasted on top of household products.

The least efficient ways to get test images are as follows:

1. Find a book and type out the first few paragraphs in any word processing document. Then, print it on a piece of A4 paper and scan it as pdf or any other image format.
2. Practice your handwriting to write as if the words are being typed. Earn sufficient money to purchase a high-end DSLR or a phone with high quality camera. Take a screenshot and transfer it to your computer.
3. Study up on the basic of pixels to fill up a 128x128 canvas with blocks of characters. If you feel that it is too time-consuming, consider take up some programming and algorithm classes to write some codes that automate the pixel-filling process.

I saved the following images as test images:

- A receipt
- An abstract from a published paper
- Introduction from a book
- Code snippet
- A few paragraphs from novels (Chinese and Japanese)
- A few Chinese emoticons

## Pre-processing

Most of the images required some form or pre-processing to improve the accuracy. Check out the following [link](https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality) to find out more on how to improve the image quality. A few important notes to be taken into account for the best accuracy:

- dark text on white background
- black and white image
- remove alpha channel (save image as jpeg/jpg instead of png)
- fine-tuning via **psm** parameters (Page Segmentations Mode)

  Page Segmentation Mode will be discussed later, in the next section. We will start with converting a image into black and white. Given the following image:cs

![Image for post](https://miro.medium.com/max/2702/1*1H0hDEAh0CafuhBDMqWJrg.png)

Sample code snippet from my Notebook

If you are using Jupyter Notebook, you can type the following code and press Shift+Enter to execute it:

``` python
from PIL import Image
column = Image.open('code.jpg')
gray = column.convert('L')
blackwhite = gray.point(lambda x: 0 if x < 200 else 255, '1')
blackwhite.save("code_bw.jpg")
```

- **Image.open(‘code.jpg’)**: **code.jpg** is the name of the file. Modify this according to the name of the input file.
- **PIL**: refers to the old version of Pillow. You only need to install Pillow and you will be able to import Image module. Do not install both Pillow and PIL.
- **column.convert(‘L’)**: **L** refers to greyscale mode. Other available options include **RGB** and **CMYK**
- **x < 200 else 255**: fine-tune the value of 200 to any other values range from 0 to 255. Check the output file to determine the appropriate value.

If you are using command-line to call a Python file. Remember to change the input file and import sys:

![Image for post](https://miro.medium.com/max/2702/1*Lv9J4ltTgs4sJedQ9z_21Q.png)

Black and white version of the code snippet

Feel free to try out other image processing methods to improve the quality of your image. Once you are done with it, let’s move on to the next section.

# 3. Usage and API calls

## Using with-statement for single image

You can use with-statement to initialize the object and `GetUTF8Text()` to get the result. This method is being referred as context manager. If you are not using with-statement, `api.End()` should be explicitly called when it’s no longer needed. Refer to the example below for manual handling for single image.

``` python
from tesserocr import PyTessBaseAPI
with PyTessBaseAPI() as api:
    api.SetImageFile('sample.jpg')
    print(api.GetUTF8Text())
```

If you encounter the following error during the call, it means the program could not locate the language data files (**tessdata** folder).

```
RuntimeError: Failed to init API, possibly an invalid tessdata path:
```

You can solve this by providing the path as argument during the initialization. You can even specify the language used — as you can see in this example (check the part highlighted in bold):

```
from tesserocr import PyTessBaseAPIwith PyTessBaseAPI(path='C:/path/to/tessdata/.', lang='eng') as api:
    api.SetImageFile('sample.jpg')
    print(api.GetUTF8Text())
```

## Using manual handling for single image

Although the recommended method is via context manager, you can still initialize it as object manually:

``` python
from tesserocr import PyTessBaseAPI
api = PyTessBaseAPI(path='C:/path/to/tessdata/.', lang='eng')
try:
    api.SetImageFile('sample.jpg')
    print(api.GetUTF8Text())
finally:
    api.End()
```

## Getting confidence value for each word

PyTessBaseAPI has several other tesseract methods that can be called. This include getting the tesseract version or even the confidence value for each word. Refer to the [*tesserorc.pyx*](https://github.com/sirfz/tesserocr/blob/master/tesserocr.pyx) file for more information. To get the word confidence, you can simply use the **AllWordConfidences()** function:

``` python
from tesserocr import PyTessBaseAPI
with PyTessBaseAPI(path='C:/path/to/tessdata/.', lang='eng') as api:
    print(api.GetUTF8Text())
    print(api.AllWordConfidences())
```

You should get a list of integers ranging from 0(worst) to 100(best) such as the results below (each score represent one word):

``` python
[87, 55, 55, 39, 88, 70, 31, 60, 18, 18, 71]
```

## Gettting all available languages

There is also a function to get all available languages: `**GetAvailableLanguages()**`. You can use the output as reference for the **lang** parameter.

``` python
from tesserocr import PyTessBaseAPI
with PyTessBaseAPI(path='C:/path/to/tessdata/.', lang='eng') as api:
    print(api.GetAvailableLanguages())
```

The output that follows depends on the number of language data files that you have in the **tessdata** folder:

``` python
['afr', 'amh', 'ara', 'asm', 'aze', 'aze_cyrl', 'bel', 'ben', 'bod', 'bos', 'bre', 'bul', 'cat', 'ceb', 'ces', 'chi_sim', 'chi_sim_vert', 'chi_tra', 'chi_tra_vert', 'chr', 'cos', 'cym', 'dan', 'dan_frak', 'deu', 'deu_frak', 'div', 'dzo', 'ell', 'eng', 'enm', 'epo', 'equ', 'est', 'eus', 'fao', 'fas', 'fil', 'fin', 'fra', 'frk', 'frm', 'fry', 'gla', 'gle', 'glg', 'grc', 'guj', 'hat', 'heb', 'hin', 'hrv', 'hun', 'hye', 'iku', 'ind', 'isl', 'ita', 'ita_old', 'jav', 'jpn', 'jpn_vert', 'kan', 'kat', 'kat_old', 'kaz', 'khm', 'kir', 'kmr', 'kor', 'kor_vert', 'lao', 'lat', 'lav', 'lit', 'ltz', 'mal', 'mar', 'mkd', 'mlt', 'mon', 'mri', 'msa', 'mya', 'nep', 'nld', 'nor', 'oci', 'ori', 'osd', 'pan', 'pol', 'por', 'pus', 'que', 'ron', 'rus', 'san', 'script/Arabic', 'script/Armenian', 'script/Bengali', 'script/Canadian_Aboriginal', 'script/Cherokee', 'script/Cyrillic', 'script/Devanagari', 'script/Ethiopic', 'script/Fraktur', 'script/Georgian', 'script/Greek', 'script/Gujarati', 'script/Gurmukhi', 'script/HanS', 'script/HanS_vert', 'script/HanT', 'script/HanT_vert', 'script/Hangul', 'script/Hangul_vert', 'script/Hebrew', 'script/Japanese', 'script/Japanese_vert', 'script/Kannada', 'script/Khmer', 'script/Lao', 'script/Latin', 'script/Malayalam', 'script/Myanmar', 'script/Oriya', 'script/Sinhala', 'script/Syriac', 'script/Tamil', 'script/Telugu', 'script/Thaana', 'script/Thai', 'script/Tibetan', 'script/Vietnamese', 'sin', 'slk', 'slk_frak', 'slv', 'snd', 'spa', 'spa_old', 'sqi', 'srp', 'srp_latn', 'sun', 'swa', 'swe', 'syr', 'tam', 'tat', 'tel', 'tgk', 'tgl', 'tha', 'tir', 'ton', 'tur', 'uig', 'ukr', 'urd', 'uzb', 'uzb_cyrl', 'vie', 'yid', 'yor']
```

## Using with-statement for multiple images

You can use a list to store the path to each images and call a for loop to loop each one of them.

``` python
from tesserocr import PyTessBaseAPI
images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg', 'sample4.jpg']
with PyTessBaseAPI(path='C:/path/to/tessdata/.', lang='eng') as api:
    for img in images:
        api.SetImageFile(img)
        print(api.GetUTF8Text())
```

tesserocr provides us with a lot of helper functions that can be used with threading to concurrently process multiple images. This method is highly efficient and should be used whenever possible. To process a single image, we can use the helper function without the need to initialize PyTessBaseAPI.

``` python
import tesserocr
images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg', 'sample4.jpg']
for img in images:
    print(tesserocr.file_to_text(img， lang='eng', path='C:/path/to/tessdata/.')
```

Other available helper functions include:

``` python
# print tesseract-ocr version
print(tesserocr.tesseract_version())
# prints tessdata path and list of available languages
print(tesserocr.get_languages()
# print ocr text from image
image = Image.open('sample.jpg')
print(tesserocr.image_to_text(image))
```

If you would like to know more about other available API calls, check the [*tesserocr.pyx*](https://github.com/sirfz/tesserocr/blob/master/tesserocr.pyx) file. Let’s move on to the next section.

# 4. Fine-tuning

In this section we will be exploring how to fine-tune tesserocr to detect different languages and setting different PSMs (Page Segmentation Mode).

## Setting other languages

You can change the language by specifying the **lang** parameter during initialization. For example, to change the language from English to Simplified Chinese, just modify eng to chi_sim, as follows:

```
lang='eng'
lang='chi_sim'
```

In fact, you can specify more than one language. Simply pipe it with a + sign. Note that the order is important as it will affects the accuracy of the results:

```
lang='eng+chi_sim'
lang='jpn+chi_tra'
```

Certain languages, such as Japanese and Chinese, have another separate category to recognize vertical text:

``` python
lang='chi_sim_vert'
lang='jpn_vert'
```

Refer to the following code on how to change the language during initialization:

``` python
# via PyTessBaseAPI during initialization
with PyTessBaseAPI(path='C:/path/to/tessdata/.', lang='chi_sim') as api:
# via helper function to get the text of single image
tesserocr.file_to_text(img， lang='eng+chi_sim', path='C:/path/to/tessdata/.'
```

## Setting the Page Segmentation Mode

During initialization, you can set another parameter called **psm,** which refers to how the model is going to treat the image. It will have an effect on the accuracy, depending on how you set it. It accepts the PSM enumeration. The list is as follows:

- 0 : `OSD_ONLY` Orientation and script detection only.
- 1 : `AUTO_OSD` Automatic page segmentation with orientation and script detection. (OSD)
- 2 : `AUTO_ONLY` Automatic page segmentation, but no OSD, or OCR.
- 3 : `AUTO` Fully automatic page segmentation, but no OSD. (default mode for tesserocr)
- 4 : `SINGLE_COLUMN`-Assume a single column of text of variable sizes.
- 5 : `SINGLE_BLOCK_VERT_TEXT`-Assume a single uniform block of vertically aligned text.
- 6 : `SINGLE_BLOCK`-Assume a single uniform block of text.
- 7 : `SINGLE_LINE`-Treat the image as a single text line.
- 8 : `SINGLE_WORD`-Treat the image as a single word.
- 9 : `CIRCLE_WORD`-Treat the image as a single word in a circle.
- 10 : `SINGLE_CHAR`-Treat the image as a single character.
- 11 : `SPARSE_TEXT`-Find as much text as possible in no particular order.
- 12 : `SPARSE_TEXT_OSD`-Sparse text with orientation and script detection
- 13 : `RAW_LINE`-Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

You can specify it in the code as follows:

``` python
with PyTessBaseAPI(path='C:path/to/tessdata/.', psm=PSM.OSD_ONLY) as api:
```

If you have issues detecting the text, try to improve the image or play around with the PSM values.

Next up, I will share some interesting results I obtained.

# 5. Results

Here are the results from my experiment running tesserocr.

## Chinese emoticons (表情包)

The images is the input file used while the caption is the results.

![Image for post](https://miro.medium.com/max/450/1*gqZwYR6R94h_KUuLcpfD0g.png)

是 的 没 错

![Image for post](https://miro.medium.com/max/450/1*Am1YGPPYl8Sm185XJ7XuvA.jpeg)

你 巳 经 超 过 1 尘
没 理 你 的 小 宝 宝 了

![Image for post](https://miro.medium.com/max/534/1*sM3ltXRLR03iVzAg3HN1jA.jpeg)

后 退 , 我 要 开 始 装 逼 了

The result is pretty good when the text are in one line. Also note that the emoticons have black text against white background. I did try white text overlaid on a scene from an animation (a colored scene, without any image pre-processing). The results were quite bad.

## English digital text

The images is the input file and the caption is the category. Results are shown after the image.

![Image for post](https://miro.medium.com/max/600/1*cBrFQAZLfsLCKovNRlV6xg.jpeg)

Receipt

```
THE SUNCADIA RESORT
TC GOLF HOUSE

1103 CLAIRE 7

1/5 1275 1 68E.1
SARK
JUNOS' 11 10: 16AH

35 HEINEKEN 157.50
35 COORS LT 175.00
12 GREY GOCSE 120.00
7 BUD LIGHT 35.00
7 BAJA CHICKEN 84.00
2 RANCHER 24.00
1 CLASSIC 8.00
1 SALMON BLT 13.00
Y DRIVER 12,00
6 CORONA 36.00
2 7-UP 4.50

Subtotal 669.00

Tax 53.52

3:36 Asnt Due $722 .52
FOR HOTEL GUEST ROOM CHARGE ONLY
Gratuity
```

![Image for post](https://miro.medium.com/max/1450/1*EbYf1PezBT1NF2YmVRO8zA.png)

E-book

```
ABOUT ESSENTIALS OF LINGUISTICS

 

This Open Educational Resource (OER) brings together Open Access content from
around the web and enhances it with dynamic video lectures about the core areas of
theoretical linguistics (phonetics, phonology, morphology, syntax, and semantics),
supplemented with discussion of psycholinguistic and neurolinguistic findings. Essentials
of Linguisticsis suitable for any beginning learner of linguistics but is primarily aimed at
the Canadian learner, focusing on Canadian English for learning phonetic transcription,
and discussing the status of Indigenous languages in Canada. Drawing on best practices
for instructional design, Essentials of Linguistics is suitable for blended classes, traditional
lecture classes, and for self-directed learning. No prior knowledge of linguistics is
required.

TO THE STUDENT

 

Your instructor might assign some parts or all of this OER to support your learning, or
youmay choose to use it to teach yourself introductory linguistics. You might decide to
read the textbook straight through and watch the videos in order, or you might select
specific topics that are of particular interest to you. However you use the OER, we
recommend that you begin with Chapter 1, which provides fundamentals for the rest of
the topics. You will also find that if you complete the quizzes and attempt the exercises,
you'll achieve a better understanding of the material in each chapter.
```

![Image for post](https://miro.medium.com/max/1328/1*n5hnYJQMnw_ytWM6ZVwUfA.png)

Abstract of a scientific paper

```
Abstract

Natural language processing tasks, such as ques-
tion answering, machine translation, reading com-
prehension, and summarization, are typically
approached with supervised learning on task-
specific datasets. We demonstrate that language
models begin to learn these tasks without any ex-
plicit supervision when trained on a new dataset
of millions of webpages called WebText. When
conditioned on a document plus questions, the an-
swers generated by the language model reach 55
F1 on the CoQA dataset - matching or exceeding
the performance of 3 out of 4 baseline systems
without using the 127,000+ training examples.
The capacity of the language model is essential
to the success of zero-shot task transfer and in-
creasing it improves performance in a log-linear
```

![Image for post](https://miro.medium.com/max/2702/1*Lv9J4ltTgs4sJedQ9z_21Q.png)

Code snippet

``` python
from tesserocr import PyTessBaseAPI

images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg', 'sample4.jpg', 'sample5.jpg',

'sample6.jpg', 'sample7.jpg']
images2 = ['examplel.png', 'example2.png', 'example3.png', 'example4.png']

with PyTessBaseAPI(path='C:/Users/wfng/test/tesserocr-master/tessdata/.', lang='eng+chi_sim') as api:
  for img in images:
    api.SetImageFile(img)
    print(api.GetUTF8Text())
    print(api.AllWordConfidences())
```

Generally results are excellent, except for some issues with formatting, spaces and line breaks.

# 6. Conclusion

This is the end of our tutorial on using tesserocr to recognize digital words in images. Although the results are promising, it’s a lot of work to create a pipeline for an actual use case. This includes image pre-processing, as well as text post-processing.

For example, to automate the auto-filling of an identity card for registration, or a receipt paper for compensation filling, there is a simple application or web service that accepts an image input. First, the application needs to crop the image and convert it into a black and white image. Then, it will pass the modified image for character recognition via tesserocr. The output text will be further processed to identify the necessary data and saved to the database. A simple feedback will be forwarded to the users, indicating that the process has been completed successfully.

Regardless of the work involved now, this technology is here to stay. Whoever is willing to spend time and resources deploying it will benefit from it. Where there is a will, there is a way!

# Reference

1. https://github.com/tesseract-ocr/tesseract
2. https://github.com/simonflueckiger/tesserocr-windows_build/releases
3. https://github.com/tesseract-ocr/tessdata
4. https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality