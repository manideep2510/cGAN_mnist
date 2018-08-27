# cGAN_MNIST
Generation of MNIST like digits using [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf).

## About

This is implementation of the paper, "[Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)" by Mehdi Mirza, Simon Osindero.

This code is implemented using [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) frameworks.

## Files

- [`cgan_mnist.py`](cgan_mnist.py) : This is the code for python implementation of [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)
- [`plots`](plots) : Loss plots for different number of total epochs
- [`images_generated`](images_generated) : Generated MNIST like digits at every sample interval (200) when training the model for 20000 epochs
- [`images_generated_5000epochs`](images_generated_5000epochs) : Generated MNIST like digits at every sample interval (50) when training the model for 5000 epochs

## Usage

Clone the repository, change your present working directory to the cloned directory, Now create a now folder in this directory named `generated` to save the generated digits after every sampled interval and now train the model. Below comands accomplishes these steps.

```
$ git clone https://github.com/manideep2510/cGAN_mnist.git
$ cd cGAN_mnist
$ mkdir generated
$ python cgan_mnist.py
```

## Let's understand what are Conditional GANs!

### What are Generative Adversarial Networks?

[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) or in short GANs are a type of generative models which can generate data which resembles the training data by learning the probability distribution of the training data through two-player minimax game between two networks namely Generator and Discriminator.

Generator try to learn the probability distribution of the training data and generate outputs which resemble the training data. Discriminator on the other hand discriminator takes the generated output from the generator and it predict whether the generated output is from the training data or not. 

The discriminator will try to tell that the generated output is not from trainng data distribution and the generator will try to fool the discriminator by generating realistic outputs close to input data distribution.

The [meaning](https://dictionary.cambridge.org/dictionary/english/adversarial) of the term adveserial means "opposing or disagreeing with each other"

Here the generator and discriminator are disagreeing with each other by generator trying to fool the discriminator and the discriminator trying to discriminate the generator.

The generator tries to generate outputs similar to training data so the generator tries to decrease it's cost but the discriminator will try to tell that the generated data is not from the input distribution, hence the discriminator tries to increase it's cost. This is why the generator and discriminator are said to be in a *two-player minimax game*.

## References

- [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf), Mehdi Mirza, Simon Osindero
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- [Keras: The Python Deep Learning library](https://keras.io/)
- [Tensorflow: An open source machine learning framework for everyone](https://www.tensorflow.org/)
