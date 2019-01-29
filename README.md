# Quick Draw: Catch doodle! 🐶🐼🐷
: *Image Classification with 50 animal drawings from Quick Draw game data at Google*

<br>

## ***1. What is Quick Draw?***
"Quick, Draw!" was released as an experimental game to educate the public in a playful way about how AI works. The game prompts users to draw an image depicting a certain category, such as ”banana,” “table,” etc. The game generated more than 1B drawings, of which a subset was publicly released as the basis for this competition’s training set. That subset contains 50M drawings encompassing 340 label categories. More details can be found [here](https://towardsdatascience.com/quick-draw-the-worlds-largest-doodle-dataset-823c22ffce6b).

![img](https://github.com/jjone36/Doodle/blob/master/img.png)

This project is for building an image classifier model that can handle noisy and sometimes incomplete drawings and perform well on classifying 50 different animals. Starting from a simple CNN as a baseline, I used Residual Net and VGG19. I choose these models because they are covered in Andrew Ng's DL specialization course On Coursera. I wanted to have a time to practice what I learnt in class. The final score was ~~ of the accuracy.

<br>

* **Project Date:** Jan, 2019

* **Applied skills:** Image Processing and visualization. Parallel Computation with Dask. Image Classification with CNN and sequence model.

* **Publication:** *(Coming soon!)*

<br>

## ***2. File Details***
- **[Doodle.ipynb](https://github.com/jjone36/Doodle/blob/master/Doodle.ipynb)** : Image processing and visualizing the drawings. Building a simple CNN as a baseline.

- **[1. Preprocessing.py](https://github.com/jjone36/Doodle/blob/master/1.%20Preprocessing.py)** : 1. data preprocessing
- **[2. Baseline.py](https://github.com/jjone36/Doodle/blob/master/2.%20Baseline.py)** : 2. Baseline Modeling
- **[3-1. ResNet50.py](https://github.com/jjone36/Doodle/blob/master/3-1.ResNet50.py)** : 3-1. Bench Mark modeling (ResNet50)
- **[3-2. VGG19.py](https://github.com/jjone36/Doodle/blob/master/3-2.VGG19.py)** : 3-2. Bench Mark modeling (VGG19)
- **[doodle.zip](https://github.com/jjone36/Doodle/blob/master/doodle.zip)** : The dataset extracted only the animal csv files

<br>
