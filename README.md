# Flappy_Bird

[Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird)  is an arcade-style game in which the player controls the bird which moves persistently to the right. The player is tasked with navigating bird through pairs of pipes that have equally sized gaps placed at random heights. Bird automatically descends and only ascends when the player says to ascend.

## Approach

I have used two different approaches to play flappy bird which is built using pygame.

* Using **Advantage Actor Critic (A2C)** which is a model-free, on-policy, reinforcement learning algorithm.
* And  also using **NeuroEvolution of Augmenting Topologies (NEAT)** which belongs to family of NeuroEvolution algorithms. I have used [neat-python](https://pypi.org/project/neat-python/) library to implement it.

## Results

| <div align="center"><img src="images/a2c.gif"/></div> | <div align="center"><img src="images/neat.gif"/></div> |
| ----------------------------------------------------- | ------------------------------------------------------ |
| <div align="center"><small><i>A2C</i></small></div>   | <div align="center"><small><i>NEAT</i></small></div>   |

Using A2C, agent started playing well after 500 episodes and using NEAT, agents started playing well after 5 Generations.

<div align="center"><small><a href="https://github.com/vstark21">&copy V I S H W A S</a></small></div>