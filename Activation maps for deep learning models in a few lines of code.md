# Activation maps for deep learning models in a few lines of code

## We illustrate how to show the activation maps of various layers in a deep CNN model with just a couple of lines of code.



![Image for post](https://miro.medium.com/max/2152/1*Oa_ERwveuGHc4hG3LbDm3A.png)

**Update**: This story received a silver badge from KDNuggets as the most shared story of October.

![Image for post](https://miro.medium.com/max/1244/1*Cfad6yRZk7t8yZoTDcHqAg.png)

https://www.kdnuggets.com/2019/10/activation-maps-deep-learning-models-lines-code.html

# Deep learning has a bad rep: ‘black-box’

**D**eep **L**earning (DL) models are [revolutionizing the business and technology world with jaw-dropping performances](https://tryolabs.com/blog/2018/12/19/major-advancements-deep-learning-2018/) in one application area after another — image classification, object detection, object tracking, pose recognition, video analytics, synthetic picture generation — just to name a few.

However, they are like anything but classical **M**achine **L**earning (ML) algorithms/techniques. DL models use millions of parameters and create extremely complex and highly nonlinear internal representations of the images or datasets that are fed to these models.

They are, therefore, often called the [**perfect black-box ML techniques**](https://www.wired.com/story/inside-black-box-of-neural-network/). We can get highly accurate predictions from them after we train them with large datasets, but [**we have little hope of understanding the internal features and representations**](https://www.technologyreview.com/s/604087/the-dark-secret-at-the-heart-of-ai/) of the data that a model uses to classify a particular image into a category.

![Image for post](https://miro.medium.com/max/1940/1*5z9lrpjfXak7Da2tCU_WfQ.png)

**Source**: [CMU ML blog](https://blog.ml.cmu.edu/2019/05/17/explaining-a-black-box-using-deep-variational-information-bottleneck-approach/)

> Black-box problem of deep learning — predictive power without an intuitive and easy-to-follow explanation.

This does not bode well because [we, humans, are visual creatures](https://www.seyens.com/humans-are-visual-creatures/). Millions of years of evolution have gifted us an [amazingly complex pair of eyes](https://www.relativelyinteresting.com/irreducible-complexity-intelligent-design-evolution-and-the-eye/) and an even more complex [visual cortex](https://www.neuroscientificallychallenged.com/blog/know-your-brain-primary-visual-cortex), and we use those organs for making sense of the world.

![Image for post](https://miro.medium.com/max/2560/1*rP-r967AD2zrlkikEoQJiA.png)

**Source**: Wikimedia

The scientific process starts with observation, and that is almost always synonymous with vision. In business, only what we can observe and measure, we can control and manage effectively.

Seeing/observing is how we start to [make mental models of worldly phenomena](https://medium.com/personal-growth/mental-models-898f70438075), classify objects around us, separate a friend from a foe, love, work, and play.

> Visualization helps a lot. Especially, for deep learning.

Therefore, a ‘black box’ DL model, where we cannot visualize the inner workings, often draws some criticism.

# Activation maps

Among various deep learning architectures, perhaps the most prominent one is the so-called **C**onvolutional **N**eural **N**etwork (CNN). [It has emerged as the workhorse for analyzing high-dimensional, unstructured data](https://www.flatworldsolutions.com/data-science/articles/7-applications-of-convolutional-neural-networks.php) — image, text, or audio — which has traditionally posed severe challenges for classical ML (non-deep-learning) or hand-crafted (non-ML) algorithms.

Several approaches for understanding and visualizing CNN have been developed in the[ literature](https://arxiv.org/pdf/1806.00069.pdf), partly as a response to the common criticism that the learned internal features in a CNN are not interpretable.

**The most straight-forward visualization technique is to show the activations** of the network during the forward pass.

> So, what are activation anyway?

At a simple level, activation functions help decide whether a neuron should be activated. This helps determine whether the information that the neuron is receiving is relevant for the input. The activation function is a non-linear transformation that happens over an input signal, and the transformed output is sent to the next neuron.

If you want to understand what precisely, these activations mean, and why are they placed in the neural net architecture in the first place, check out this article,

[Fundamentals of Deep Learning - Activation Functions and When to Use Them?Introduction Internet provides access to a plethora of information today. Whatever we need is just a Google (search)…www.analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/)

Below is a fantastic video by the renowned data scientist [Brandon Rohrer](https://brohrer.github.io/blog.html) about the basic mechanism of a CNN i.e. how a given input (say a two-dimensional image) is processed layer by layer. At each layer, the output is generated by passing the transformed input through an activation function.

<iframe src="https://cdn.embedly.com/widgets/media.html?src=https%3A%2F%2Fwww.youtube.com%2Fembed%2FILsA4nyG7I0%3Ffeature%3Doembed&amp;url=http%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DILsA4nyG7I0&amp;image=https%3A%2F%2Fi.ytimg.com%2Fvi%2FILsA4nyG7I0%2Fhqdefault.jpg&amp;key=a19fcc184b9711e1b4764040d3dc5c07&amp;type=text%2Fhtml&amp;schema=youtube" allowfullscreen="" frameborder="0" height="480" width="854" title="How Deep Neural Networks Work" class="s t u ib ai" scrolling="auto" style="box-sizing: inherit; position: absolute; top: 0px; left: 0px; width: 680px; height: 382.188px;"></iframe>

















Activation maps are just a visual representation of these activation numbers at various layers of the network as a given image progresses through as a result of various linear algebraic operations.

For ReLU activation based networks, the activations usually start out looking relatively blobby and dense, but as the training progresses the activations usually become more sparse and localized. One design pitfall that can be easily caught with this visualization is that some activation maps may be all zero for many different inputs, which can indicate *dead* filters and can be a symptom of high learning rates.

> Activation maps are just a visual representation of these activation numbers at various layers of the network.

Sounds good. **But visualizing these activation maps is a non-trivial task**, even after you have trained your neural net well and are making predictions out of it.

> How do you easily visualize and show these activation maps for a reasonably complicated CNN with just a few lines of code?

# Activation maps with a few lines of code

The whole [**Jupyter notebook is here**](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keract-activation.ipynb). Feel free to fork and expand (and leave a star for the repository if you like it).

## A compact function and a nice little library

I showed previously in an article, how to write a single compact function for obtaining a fully trained CNN model by reading image files one by one automatically from your disk, by utilizing some amazing utility methods and classes offered by Keras library.

**Do check out this article, because, without it, you cannot train arbitrary models with arbitrary image datasets in a compact manner, as described in this article**.

[A single function to streamline image classification with KerasWe show, how to construct a single, generalized, utility function to pull images automatically from a directory and…towardsdatascience.com](https://towardsdatascience.com/a-single-function-to-streamline-image-classification-with-keras-bd04f5cfe6df)

Next, we use this function along with a [nice little library called **Keract**](https://github.com/philipperemy/keract), which makes the visualization of activation maps super easy. It is a high-level accessory library to Keras library to show useful heatmaps and activation maps on various layers of a neural network.

![Image for post](https://miro.medium.com/max/2292/1*nKNEE3BusnZlHeUcsV7d2Q.png)

Therefore, for this code, we need to use a couple of utility functions from my `utils.DL_utils` module - `train_CNN_keras` and `preprocess_image` to make a random RGB image compatible for generating the activation maps (these were described in the article mentioned above).

[**Here is the Python module —** ](https://raw.githubusercontent.com/tirthajyoti/Deep-learning-with-Python/master/Notebooks/utils/DL_utils.py)`**DL_utils.py**`. You can store in your local drive and import the functions as usual.

## The dataset

For training, we are using the famous **Caltech-101 dataset** from http://www.vision.caltech.edu/Image_Datasets/Caltech101/. This dataset was somewhat a **precursor to the** [**ImageNet database**](http://image-net.org/), which is the current gold standard for image classification data repository.

![Image for post](https://miro.medium.com/max/3602/1*ZyIcNlplNO4jClEUTsJbRA.png)

It is an image dataset of diverse types of objects belonging to 101 categories. There are about 40 to 800 images per category. Most categories have about 50 images. The size of each image is roughly 300 x 200 pixels.

However, we are training only with 5 categories of images — *crab, cup, brain, camera*, and *chair*.

This is just a random choice for this demo, feel free to choose your own categories.

## Training the model

Training is done in a few lines of code only.

<iframe src="https://towardsdatascience.com/media/4dae5cd11013207e6e0c29ca4e148881" allowfullscreen="" frameborder="0" height="263" width="680" title="keract-1" class="s t u ib ai" scrolling="auto" style="box-sizing: inherit; position: absolute; top: 0px; left: 0px; width: 680px; height: 263px;"></iframe>

## A random image of a human brain downloaded from the internet

For generating the activations, we download a random image of a human brain from the internet.

![Image for post](https://miro.medium.com/max/972/1*ZMr0xXmUO6CYV8NBvRIr0Q.png)

## Generate the activations (a dictionary)

Then, another couple of lines of code to generate the activation.

``` python
from keract import display_activations
# The image path
img_path = '../images/brain-1.jpg'
# Preprocessing the image for the model
x = preprocess_image(img_path=img_path,model=model,resize=target_size)
# Generate the activations 
activations = get_activations(model, x)
```

We get back a dictionary with layer names as the keys and Numpy arrays as the values corresponding to the activations. Below an illustration is shown where the activation arrays are shown to have varying lengths corresponding to the size of the filter maps of that particular convolutional layer.

![Image for post](https://miro.medium.com/max/874/1*JdfXvhU7CXWbErhGTEKX7A.png)

## Display the activations

Again, a single line of code,

```
display_activations(activations, save=False)
```

We get to see activation maps layer by layer. Here is the first convolutional layer (**16 images corresponding to the 16 filters**)

![Image for post](https://miro.medium.com/max/1378/1*eINNAshsPWFV3h0yL85aAQ.png)

And, here is layer number 2 (**32 images corresponding to the 32filters**)

![Image for post](https://miro.medium.com/max/1390/1*kBvaEEPkOQOPTCRGvZhSpA.png)

We have 5 convolutional layers (followed by Max pooling layers) in this model, and therefore, we get back 10 sets of images. For brevity, I am not showing the rest but you can see them all in my Github repo here.

## Heatmaps

You can also show the activations as heatmaps.

```
display_heatmaps(activations, x, save=False)
```

![Image for post](https://miro.medium.com/max/1360/1*jdSMqGD7c0Uwb3CtpV0fow.png)

# Update: Quiver

After writing this article last week, I found out about another beautiful library for activation visualization called [**Quiver**](https://github.com/keplr-io/quiver). However, this one is built on the Python microserver framework Flask and displays the activation maps on a browser port rather than inside your Jupyter Notebook.

They also need a trained Keras model as input. So, you can easily use the compact function described in this article (from my [DL_uitls module](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/utils/DL_utils.py)) and try this library for interactive visualization of activation maps.

![Image for post](https://miro.medium.com/max/2560/1*bmzltzIAbv_KoT81GqEPqw.gif)

# Summary

That’s it, for now.

The whole [**Jupyter notebook is here**](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keract-activation.ipynb).

We showed, how using only a few lines of code (utilizing compact functions from a special module and a nice little accessory library to Keras) we can train a CNN, generate activation maps, and display them layer by layer — from scratch.

This gives you the ability to train CNN models (simple to complex) from any image dataset (as long as you can arrange it in a simple format) and look inside their guts for any test image you want.

For more of such hands-on tutorials, [**check my Deep-Learning-with-Python Github repo**](https://github.com/tirthajyoti/Deep-learning-with-Python).