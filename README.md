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

Clone the repository by trying the below command in your terminal.

```git clone https://github.com/manideep2510/cGAN_mnist.git```

Then change you present working directory to the cloned directory with the below command.

```cd cGAN_mnist```

Now create a now folder in this directory named `generated` to save the generated digits after every sampled interval with the following command.

```mkdir generated```

To train the model, type the following command in the terminal.

```python cgan_mnist.py```

## References

- [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf), Mehdi Mirza, Simon Osindero
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- [Keras: The Python Deep Learning library](https://keras.io/)
- [Tensorflow: An open source machine learning framework for everyone](https://www.tensorflow.org/)
