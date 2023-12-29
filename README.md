# marl_resource_allocation
**Read this in other languages: [中文](README_zh.md).**
#### introduce
qmix,vnd,coma to allocate resources. Because some Chinese notes may be marked for personal use, please delete them if not necessary after clone.

#### Software architecture

The /agent directory is the agent implementation.

The /common directory is an implementation of common methods such as utils.

The /data directory is the implementation of the experimental data storage location.

The /env directory is the implementation of the environment Settings.

The /generate directory is the implementation of the data enhancement extension.

The /model directory is where the model is saved.

The /network directory is the implementation of the network structure.

The /policy directory is the implementation of the algorithm.

The /requirement directory is an implementation of the user's requirement.

The /result directory is the location where data such as images of experiment results are saved.


#### Installation tutorial

1.  First clone the repository locally
2.  Install dependencies in your home directory according to requirements.txt（There are dependencies that may not be used, such as StarCraft 2's dependencies, even if they are not installed）
3.  You need to modify the parameters in/common/[argument.py](common%2Fargument.py)，Including learning rate, total step size, etc.And then you can run it through [main.py](main.py). 


#### Partial result presentation

1.  Experimental results of different algorithms
![](D:\pycharm\pythonProject\marl_resource_allocation\result1.png)
2.  Comparison of training process under laziness reward
![](D:\pycharm\pythonProject\marl_resource_allocation\result2.png)
3.  qmix algorithm based on the mechanism (1 million steps training)
![](D:\pycharm\pythonProject\marl_resource_allocation\result3.png)

#### statement

This code provides support for the paper "Online resource allocation model with time slot based on qmix". You may not use the code for your own paper submission without permission. Failure to do so will result in accountability.


