
This project is an implementation of the architecture described in the research paper - DeepONet: Learning nonlinear operators based on the universal approximation theorem of operators, authored by Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. The paper can be found [here](https://arxiv.org/pdf/1910.03193v2).

The DeepONet architecture is a deep learning framework that can learn nonlinear operators from data. To approximate a non linear operator, the network needs to take the input of any desired function to be operated on, and output the value of the new function at a desired point. One way to input a function is to sample the function at a set of points over the domain.

Following is the architecture of the DeepONet model:

![DeepONet Architecture](/assets/architecture.png)


We were able to implement the DeepONet model with PyTorch and trained it to estimate antiderivatives of quadratic functions. The model achieved an r2 score of 0.99 on unseen test data.

The model was trained on 1000 samples of quadratic functions of the form `ax^2 + bx + c` where `a`, `b`, and `c` are randomly sampled from the range [-1, 1]. The model was trained to estimate the antiderivative of the quadratic function at a point `x`. 

Following is a schematic of the training data:

![Training Data](/assets/dataset.png)

While the above is a fairly simple example, it's a proof of concept for such models. A deeper model with more layers of neurons can generalize to more complex operators.

Following are the training metrics of the model:

![Training Metrics](/assets/training_metrics.png)