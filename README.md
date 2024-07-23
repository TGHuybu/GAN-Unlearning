# MUL-Undergrad-Thesis
*Update 2/6/2024*<br />
The basic components of GAN is neurol network, so in this early version of GAN, I try to build a CNN from scratch to classified number from the MNIST dataset

*Update 8/6/2024*<br />
We transform the source from .ipynb to .py extension in order to control the source more easier on a much bigger scale (our thesis)


*Update 23/7/2024*<br />
Brand new update
I make a new GAN due to some stability problem in the former version, this commit have address these problem :

Loss balance between D and G ( In this update, loss value of both network is seems like they're fighting against each other)
I also add some detail value of each network in the final iteration in each epoch in order to have a better assessment
Thus, some problem is occured :

When lr of G 10 times bigger than D (10^-4 and 10^-3, resp), the model colapse very fast ( after around 30 epoch )