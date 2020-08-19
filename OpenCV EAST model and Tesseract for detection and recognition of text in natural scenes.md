# OpenCV EAST model and Tesseract for detection and recognition of text in natural scenes

![Image for post](https://miro.medium.com/max/2052/1*Bc-C54f2E0YwIrAfsZ8ZVQ.png)

[Source](https://towardsdatascience.com/neural-networks-intuitions-6-east-5892f85a097)

Google has digitized books ang Google earth is using NLP to identify addresses, but how does it work exactly?

Deep learning approaches like neural networks can be used to combine the tasks of localizing text (Text detection) in an image along with understanding what the text is (Text recognition).

# ***Unstructured Text***

Text at random places in a natural scene. Sparse text, no proper row structure, complex background , at random place in the image and no standard font.

![Image for post](https://miro.medium.com/max/1164/1*d_28aRMoy_Z9kuJwHFN_yQ.png)

Unstructured Texts: Handwritten, Multiple fonts and sparse; Image source: [https://pixabay.com](https://pixabay.com/)

A lot of earlier techniques solved the OCR problem for structured text.

But these techniques didn’t properly work for a natural scene, which is sparse and has different attributes than structured data.

***In this blog, we will be focusing more on unstructured text which is a more complex problem to solve***.

As we know in the deep learning world, there is no one solution which works for all. We will be seeing multiple approaches to solve the task at hand and will work through one approach among them.

# Datasets for unstructured OCR tasks

There are lots of datasets available in English but it’s harder to find datasets for other languages. Different datasets present different tasks to be solved. Here are a few examples of datasets commonly used for machine learning OCR problems.

## SVHN dataset

The Street View House Numbers dataset contains 73257 digits for training, 26032 digits for testing, and 531131 additional as extra training data. The dataset includes 10 labels which are the digits 0–9. The dataset differs from MNIST since [SVHN](http://www.iapr-tc11.org/mediawiki/index.php?title=The_Street_View_House_Numbers_(SVHN)_Dataset) has images of house numbers with the house numbers against varying backgrounds. The dataset has bounding boxes around each digit instead of having several images of digits like in MNIST.

## Scene Text dataset

[This dataset](http://www.iapr-tc11.org/mediawiki/index.php?title=KAIST_Scene_Text_Database) consists of 3000 images in different settings (indoor and outdoor) and lighting conditions (shadow, light and night), with text in Korean and English. Some images also contain digits.

## Devanagri Character dataset

[This dataset](http://www.iapr-tc11.org/mediawiki/index.php?title=Devanagari_Character_Dataset) provides us with 1800 samples from 36 character classes obtained by 25 different native writers in the devanagri script.

And there are many others like this one for [chinese characters](http://www.iapr-tc11.org/mediawiki/index.php?title=Harbin_Institute_of_Technology_Opening_Recognition_Corpus_for_Chinese_Characters_(HIT-OR3C)), this one for [CAPTCHA](https://www.kaggle.com/fournierp/captcha-version-2-images) or this one for [handwritten words](http://ai.stanford.edu/~btaskar/ocr/).

# Text Detection :

![Image for post](https://miro.medium.com/max/604/1*nbXhfonBW-wj3CdhMNy0sA.jpeg)

[Source](https://www.learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/)

Text detection techniques required to detect the text in the image and create and bounding box around the portion of the image having text. Standard objection detection techniques will also work here.

## Sliding window technique

The bounding box can be created around the text through the sliding window technique. However, this is a computationally expensive task. In this technique, a sliding window passes through the image to detect the text in that window, like a convolutional neural network. We try with different window size to not miss the text portion with different size. There is a convolutional implementation of the sliding window which can reduce the computational time.

## Single Shot and Region based detectors

There are single-shot detection techniques like YOLO(you only look once) and region-based text detection techniques for text detection in the image.

![Image for post](https://miro.medium.com/max/3840/1*WbLERLSfnoibm4NvdRvubQ.png)

YOLO architecture: [source](https://arxiv.org/pdf/1506.02640.pdf)

YOLO is single-shot techniques as you pass the image only once to detect the text in that region, unlike the sliding window.

Region-based approach work in two steps.

First, the network proposes the region which would possibly have the test and then classify the region if it has the text or not. You can refer one of my previous [article](https://towardsdatascience.com/object-detection-using-deep-learning-approaches-an-end-to-end-theoretical-perspective-4ca27eee8a9a) to understand techniques for object detection, in our case text detection.

## EAST (Efficient accurate scene text detector)

This is a very robust deep learning method for text detection based on this [paper](https://arxiv.org/abs/1704.03155v2). It is worth mentioning as it is only a text detection method. It can find horizontal and rotated bounding boxes. It can be used in combination with any text recognition method.

The text detection pipeline in this paper has excluded redundant and intermediate steps and only has two stages.

One utilizes the fully convolutional network to directly produce word or text-line level prediction. The produced predictions which could be rotated rectangles or quadrangles are further processed through the non-maximum-suppression step to yield the final output.

![Image for post](https://miro.medium.com/max/1104/1*eNfiDMITtg7_86ZhbAhwlQ.png)

[Source](https://arxiv.org/pdf/1704.03155v2.pdf)

EAST can detect text both in images and in the video. As mentioned in the paper, it runs near real-time at 13FPS on 720p images with high text detection accuracy. Another benefit of this technique is that its implementation is available in OpenCV 3.4.2 and OpenCV 4. We will be seeing this EAST model in action along with text recognition.

# Text Recognition :

Once we have detected the bounding boxes having the text, the next step is to recognize text. There are several techniques for recognizing the text. We will be discussing some of the best techniques in the following section.

## CRNN

Convolutional Recurrent Neural Network (CRNN) is a combination of CNN, RNN, and CTC(Connectionist Temporal Classification) loss for image-based sequence recognition tasks, such as scene text recognition and OCR. The network architecture has been taken from this [paper](https://arxiv.org/abs/1507.05717) published in 2015.

![Image for post](https://miro.medium.com/max/1432/1*BCzY-Nwt3ZspD9rzzLcLjQ.png)

[Source](https://arxiv.org/pdf/1507.05717.pdf)

This neural network architecture integrates feature extraction, sequence modeling, and transcription into a unified framework. This model does not need character segmentation. The convolution neural network extracts features from the input image(text detected region). The deep bidirectional recurrent neural network predicts label sequence with some relation between the characters. The transcription layer converts the per-frame made by RNN into a label sequence. There are two modes of transcription, namely the lexicon-free and lexicon-based transcription. In the lexicon-based approach, the highest probable label sequence will be predicted.

## Tesseract

Tesseract was originally developed at Hewlett-Packard Laboratories between 1985 and 1994. In 2005, it was open-sourced by HP. As per wikipedia-*In 2006, Tesseract was considered one of the most accurate open-source OCR engines then available.*

The capability of the Tesseract was mostly limited to structured text data. It would perform quite poorly in unstructured text with significant noise. Further development in tesseract has been sponsored by Google since 2006.

Deep-learning based method performs better for the unstructured data. Tesseract 4 added deep-learning based capability with LSTM network(a kind of Recurrent Neural Network) based OCR engine which is focused on the line recognition but also supports the legacy Tesseract OCR engine of Tesseract 3 which works by recognizing character patterns. The latest stable version 4.1.0 is released on July 7, 2019. This version is significantly more accurate on the unstructured text as well.

*We will use some of the images to show both text detection with the EAST method and text recognition with Tesseract 4. Let’s see text detection and recognition in action in the following code.* The article [here](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/) proved to be a helpful resource in writing the code for this project.

``` python
#Loading packages 
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt

#Creating argument dictionary for the default arguments needed in the code. 
args = {"image":"../input/text-detection/example-images/Example-images/ex24.jpg", "east":"../input/text-detection/east_text_detection.pb", "min_confidence":0.5, "width":320, "height":320}
```

Here, I am working with essential packages. OpenCV package uses the EAST model for text detection. The tesseract package is for recognizing text in the bounding box detected for the text. Make sure you have tesseract version >= 4. There are several sources available online to guide installation of the tesseract.

Created a dictionary for the default arguments needed in the code. Let’s see what these arguments mean.

- *image: The location of the input image for text detection & recognition.*
- *east: The location of the file having the pre-trained EAST detector model.*
- *min-confidence: Min probability score for the confidence of the geometry shape predicted at the location.*
- *width: Image width should be multiple of 32 for the EAST model to work well.*
- *height: Image height should be multiple of 32 for the EAST model to work well.*

``` python
#Give location of the image to be read.

args['image']="example-image.jpg"
image = cv2.imread(args['image'])

#Saving a original image and shape
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new height and width to default 320 by using args #dictionary.  
(newW, newH) = (args["width"], args["height"])

#Calculate the ratio between original and new image for both height and weight. 
#This ratio will be used to translate bounding box location on the original image. 
rW = origW / float(newW)
rH = origH / float(newH)

# resize the original image to new dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# construct a blob from the image to forward pass it to EAST model
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
```
Loading Pre-trained EAST model and defining output layers and forward pass the image

``` python
#Give location of the image to be read.

args['image']="example-image.jpg"
image = cv2.imread(args['image'])

#Saving a original image and shape
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new height and width to default 320 by using args #dictionary.  
(newW, newH) = (args["width"], args["height"])

#Calculate the ratio between original and new image for both height and weight. 
#This ratio will be used to translate bounding box location on the original image. 
rW = origW / float(newW)
rH = origH / float(newH)

# resize the original image to new dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# construct a blob from the image to forward pass it to EAST model
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
```

Now, we have to derive the bounding boxes after applying [non-max-suppression](https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/).

``` python
# Find predictions and  apply non-maxima suppression
(boxes, confidence_val) = predictions(scores, geometry)
boxes = non_max_suppression(np.array(boxes), probs=confidence_val)

##Text Detection and Recognition 

# initialize the list of results
results = []

# loop over the bounding boxes to find the coordinate of bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	#extract the region of interest
	r = orig[startY:endY, startX:endX]

	#configuration setting to convert image to string.  
	configuration = ("-l eng --oem 1 --psm 8")
    ##This will recognize the text from the image of bounding box
	text = pytesseract.image_to_string(r, config=configuration)

	# append bbox coordinate and associated text to the list of results 
	results.append(((startX, startY, endX, endY), text))
```

We can choose the specific Tesseract configuration on the basis of our image data.

``` python
#Display the image with bounding box and recognized text
orig_image = orig.copy()

# Moving over the results and display on the image
for ((start_X, start_Y, end_X, end_Y), text) in results:
	# display the text detected by Tesseract
	print("{}\n".format(text))

	# Displaying text
	text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
	cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
		(0, 0, 255), 2)
	cv2.putText(orig_image, text, (start_X, start_Y - 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)

plt.imshow(orig_image)
plt.title('Output')
plt.show()
```
# Results :

The code uses OpenCV EAST model for text detection and tesseract for text recognition. PSM for the Tesseract has been set accordingly to the image. It is important to note that Tesseract normally requires a clear image for working well.

In our current implementation, we did not consider rotating bounding boxes due to its complexity to implement. But in the real scenario where the text is rotated, the above code will not work well. Also, whenever the image is not very clear, tesseract will have difficulty to recognize the text properly.

We can not expect the OCR model to be 100 % accurate. Still, we have achieved good results with the EAST model and Tesseract. Adding more filters for processing the image would help in improving the performance of the model.https://policy.medium.com/medium-terms-of-service-9db0094a1e0f?source=post_page-----1fa48335c4d1----------------------)