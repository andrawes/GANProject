# CapsGAN

## About
This project is a shot experiment in combining Generative Adversarial Networks with Capsule Networks [1].
Undertaken at ETH Zurich, as a short semester project, by Andrawes Al Bahou at the Computer Vision Lab (Prof. Luc Van Gool).

## Abstract
Traditional Convolutional Neural Networks (CNNs) suffer a few shortcomings. In particular, they suffer from
a difficulty in generalizing a learned representation of an object when shown in a different viewpoint or in a
new pose (given by a different rotation, scaling or translation). To learn this invariance, they must be trained
on many differing viewpoints, which is however non-trivial to achieve in practice.
In order to address these pitfalls, Hinton et al. recently make an initial attempt to propose an alternative
baptized Capsule Networks using dynamic routing. Essentially capsules are a group of neurons outputting
an activity vector, whose length is to represent the probability that the entity exits and its orientation is to
represent the instantiation parameters (such as location, scale, rotation, skewness, etc). In the context of their
proposed dynamic routing system, the capsules for low-level features are designed to predict the outputs of
higher level capsules corresponding to more abstract features. When low-level capsules agree on the predicted
outcome of a particular higher-level capsule, then their outputs are routed to this higher level-capsule.
In this experiment we explore the potential usefulness of Capsule Networks with dynamic routing under different
scenarios for generative modeling. Particularly we study their performance as discriminators in the
domain of Generative adversarial networks (GANs) by designing multiple network architectures for standard
datasets. From the study, we discover that while their use is indeed feasible, their performance is poor
on datasets with rich image semantics. In addition, we devise a novel ”inverted” Capsule Network as a
generative model for GANs, and uncover inherent instability when used to generate images. Finally, we
explore the reasons behind this and suggest a possible, (but expensive) way of solving it


## Some helpful material to understand Capsule Networks and Dynamic Routing: 
Original Paper [https://arxiv.org/pdf/1710.09829.pdf] 
Explanation Video by Aurélien Géron [https://www.youtube.com/watch?v=pPN8d0E3900] 

## References:
[1] - Dynamic Routing Between Capsules. S.Sabor, N.Frosst, G.Hinton
