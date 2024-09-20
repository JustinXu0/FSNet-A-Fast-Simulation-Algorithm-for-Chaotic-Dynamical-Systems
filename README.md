# FSNet: A Fast Simulation Algorithm for Chaotic Dynamical Systems
## 1. Introction
It is an instruction manual for my graduation project. Following the same settings as NeurVec, this method demonstrates excellent performance on the datasets, including Elastic Pend, Klink, Spring Chain.

Conventional piecewise linear activation function, such as ReLU in the following figure, cannot guarantee the existence of Lipschitz smoothness in the defined domain of the network, which leads to poor simulation performance in the simulation of chaotic systems. The following figure shows the performance of many conventional activation functions on Elastic Pend.
![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/0e34745b-069f-48cb-ad2f-c29d025a445d)

We found that performance of these activation functions are much lower than that of NeurVec, which meet the Lipschitz smoothness condition.

Observing that the exponential operation contained in the rational used by NeuroVec affects running speed of the model, we replace rational with an activation function that satisfies the Lipschitz smoothness.

$$ \phi (x) = a_0 \frac{d(sigmoid(a_1x))}{dx} + a_2 x + a_3 $$

Also, their original and derivative functions have extremely similar geometric shapes.

![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/1e022a5b-070d-4de2-a80f-c6f10a71ca5e)

During the experiment, a heatmap was drawn for the pre activation values of the feature map, and it was found that there was a regular distribution of values: the colors of each channel showed a band like distribution at the same time step, indicating that the values of different samples in the same channel were similar; The color of the same channel varies at different time steps, indicating that the distribution of pre activation values for the same channel varies at different time steps. The following figure describes the values of feature maps with a size of (TimeStep * BatchSize) * ChannelNum.

![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/e0e7004b-ef45-4217-b979-f4e4265337c7)

Naturally, we apply a channel-wise activation function to solve different values of channel-wise distribution.

Then, in order to handle chaotic problems better and encode high-frequency information, a periodic oscillation activation function,SIREN, is introduced to adapt to the instability of the solution and reduce spectral deviation.

The formula represents the channel-wise version of SIREN, N represents the number of different channels.
$$\phi (x_i) = sin(W_i x_i + b_i), i=0,1,...,N_{channel} -1$$

## 2. Ablation experiment
### 2.1 Function Replacement
![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/9abfae5e-8867-4318-86eb-592f0b71fff5)

![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/63f07b26-4c37-434a-b80a-18cc8c9e44fd)

### 2.2 Channel-wise Activation Function
![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/b1b1a2b5-4de7-443c-bbeb-d584bc9420c7)

### 2.3 SIREN
![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/f1fb292a-6757-4a01-9496-ae7e56229e0e)


## 3. Performance

Based on the above improvements, we have obtained our final method. Our method balances speed and accuracy, greatly improving the efficiency of simulation.

### 3.1 Accuracy of Prediction
Compared with NeuroVec, our method has improved simulation accuracy by **2 orders of magnitude**. 

![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/9f7a5ca3-4d1c-4497-94a3-d8b4bb4a1234)

### 3.2 Speed of Convergence

When using only 25% of the original method's data, the convergence speed is **4 times faster**.

![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/85f88839-61f2-41ca-9e3a-2096602571a5)

### 3.3 Speed of Inference

Compared to NeuroVec, the inference speed is **22% faster**, which is **150 times faster** than traditional methods.

![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/a3a96d3a-98b7-422e-84b7-28177a496ad2)

### 3.4 Stability of Trajectory

When visualizing the motion trajectory, our method predicts a **more stable path**.
![image](https://github.com/small-whirlwind/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems/assets/59130750/9abad1ca-3c0b-40db-a514-d3119e1bbe29)

## Citation
```
@article{JustinXu0,
      title={FSNet: A Fast Simulation Algorithm for Chaotic Dynamical Systems}, 
      author={Yuanfeng Xu, Shenglan Liu},
      year={2023},
      url={https://github.com/JustinXu0/A-Fast-Simulation-Algorithm-for-Chaotic-Dynamical-Systems},
      primaryClass={cs.CV}
}
```
