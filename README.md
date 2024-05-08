# Forward-and-Backward-Propagation-in-MLP
-- Yuanshan Zhang

I here provide a well-explained forward and backward propagation in MLP
## MLP Structure
A typical MLP is structured as follows:
![示例图片](images/MLP.png)
MLP basically consists of: 1. input layer 2. hidden layers 3. output layer 4. activation neurons 5. bias units. We can observe that activation neurons of two neighbor layers are connected by weights, and biases of the current layer are broadcasted to the activation neurons on the next layer. 

## Forward Propagation
Forward propagtion updates the values of activation neurons.

During the forward propagation of neurons on the $l$ th hidden layer to the $(l+1)$ th hidden layer, define the followings:
- $A^{(l)}$ is the activation matrix with $P$ neurons for each sample on the $l$ th hidden layer
- $A^{(l+1)}$ is the activation matrix with $Q$ neurons for each sample on the $(l+1)$ th hidden layer
- $W^{(l)}$ is the weight matrix connecting the activation neurons on the lth hidden layer to the activation neurons $(l+1)$ th hidden layer
- $B^{(l)}$ is the bias matrix on the lth layer, and is initialized to 1
- $\phi(Z^{(l)})$ is the activation function that calculates $A^{(l+1)}$
- $Z^{(l)}$ is the net input function that combines $A^{(l)}$ and $W^{(l)}$
- $m$ is the batch size
- $p$ is the index for a neuron on the $l$ th layer
- $q$ is the index for a neuron on the $(l+1)$ th layer

For the $i$ th sample, activate neurons on the $l$ th hidden layer are: $A_i^{(l)}=\left[a_{i1} \ a_{i2} \ldots a_{iP} \right]$

The weight vector connecting to the qth neuron on the (l+1)th hidden layer from the lth hidden layer is:


$$
W_{j=q}^{(l)}=\left[\begin{array}{c}
w_{1 q} \\
w_{2 q} \\
\vdots \\
w_{P q}
\end{array}\right]
$$

Net input function from the $l$ th layer to the $(l+1)$ th layer is:
$\begin{aligned} z_{iq}^{(l)}=(b_i+a_{i 1} w_{1 q}+\cdots+a_{i P} w_{P q})^{(l)}=b_i^{(l)} +A_i^{(l)} W_{j=q}^{(l)} \end{aligned}$

