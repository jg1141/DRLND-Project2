# DRLND-Project2 README
DRLND Project 2 - Continuous Control (Reacher environment)

---

The notebook and Python files in this repository present a solution using the Unity ML-Agents **Reacher** environment for the second project of the Udacity [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) (July 2018).

## Project Details

The *state space* for this project consists of 11 continuous 3-dimensional vectors representing the position, rotation, velocity, and angular velocities of the two parts of a virtual "reacher" arm, along with the position of the "hand" and the position and speed of the goal sphere. The goal sphere is programmed to circle around the arm (within the X-Y plane, Z is constant).

The *action space* is a vector with four numbers, clamped between -1 and 1, corresponding to the X and Z torques applicable to arm's two joints ("shoulder" and "elbow"). 

Code for the **ReacherAgent** can be viewed [here](https://github.com/Unity-Technologies/ml-agents/blob/master/UnitySDK/Assets/ML-Agents/Examples/Reacher/Scripts/ReacherAgent.cs)

The image below shows a version of the environment with 10 arms:

![Reacher Environment](reacher.png)

A video of the hands (blue) and goal spheres (green) in motion is [here](https://youtu.be/2N9EoF6pQyE).

A reward of +0.01 is provided for each step that the agent's hand is in the goal location.

The environment is considered solved when the agent has an average score of 30 or higher in 100 episodes.

## Getting Started

#### Install the Anaconda distribution of Python 3

[Anaconda Python](https://www.anaconda.com/download/#macos) installation.

#### Obtain Unity ML-Agents

Install [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

## Instructions

#### Start Jupyter and open the DRLND Project 2 notebook

```
> jupyter notebook
```

The notebook, **DRLND Project 2 - Continuous Control (Reacher environment).ipynb**, can then be opened.

The **Reacher environment** used for this project was supplied in the Udacity DRLND on-line Workspace.

Select **Cell > Run All**.

## Additional Resources

Need more links? Visit [http://bit.ly/drlndlinks](http://bit.ly/drlndlinks) to learn much more about DRLND and Reinforcement Learning.
