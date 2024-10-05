# AI Snake Game

## Project Overview

The AI Snake Game is a Python-based application that uses deep learning techniques to train a snake to play the game autonomously. Using PyTorch and a custom reinforcement learning algorithm, the snake learns to navigate the game environment, avoid collisions, and maximize its score. Currently it cannot detect its full body, so it can still collide with itself and gets into a loop.

## Features

- **Reinforcement Learning**: Utilizes a Q-learning model with a neural network to decide the snake's direction.
- **Customizable Training Episodes**: Configure the number of training episodes and batch size.
- **Real-time Training Visualization**: Visual output of the snake's movements and score as it learns.

## Prerequisites

Before you can run this project, you will need to have the following installed:

- Python 3.8 or higher
- PyTorch 1.8 or higher
- NumPy
- Pygame
