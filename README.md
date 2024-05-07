# Forward-and-Backward-Propagation-in-MLP
-- Yuanshan Zhang

I here provide a well-explained forward and backward propagation in MLP
## MLP Structure

## Forward Propagation
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
