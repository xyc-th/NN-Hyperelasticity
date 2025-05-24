from .config import *


class ConvexLinear(torch.nn.Module):
    """
    Custom linear layer with positive weights and no bias

    Initialize:
    size_in, size_out

    Inputs:
    input data

    Outputs:
    input data times softplus(trainable weight)
    """

    def __init__(self, size_in, size_out):
        super().__init__()
        weights = torch.Tensor(size_out, size_in)
        self.size_in = size_in
        self.size_out = size_out
        self.weights = torch.nn.Parameter(weights)

        # Initialize weights
        # Keeping the variance of the activation values in each layer consistent in the forward propagation,
        # thus avoiding the problem of vanishing gradients or gradient explosion
        torch.nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))

    def forward(self, x):
        z = torch.mm(x, torch.nn.functional.softplus(self.weights.t()))
        return z


class ICNN(torch.nn.Module):
    """
    Material model based on Input convex neural network

    Initialize:
    n_input:        Input layer size
    n_hidden:       Hidden layer size / number of neurons
    n_output:       Output layer size
    use_dropout:    Activte dropout during training
    dropout_rate:   Dropout probability

    Inputs:
    Deformation gradient in the form: (F11, F12, F13, F21, F22, F23, F31, F32, F33)

    Outputs:
    NN-based strain energy density (W_NN)

    """

    def __init__(self, n_input, n_hidden, n_output, use_dropout, dropout_rate, bypass=False):
        super(ICNN, self).__init__()
        # Create Module dicts for the hidden and skip-connection layers
        self.layers = torch.nn.ModuleDict()
        self.skip_layers = torch.nn.ModuleDict()
        self.depth = len(n_hidden)
        self.dropout = use_dropout
        self.p_dropout = dropout_rate
        self.bypass = bypass

        # Create the first layer of NN
        self.layers[str(0)] = torch.nn.Linear(n_input, n_hidden[0]).float()

        # Create NN with number of elements in n_hidden as depth
        for i in range(1, self.depth):
            self.layers[str(i)] = ConvexLinear(n_hidden[i - 1], n_hidden[i]).float()
            self.skip_layers[str(i)] = torch.nn.Linear(n_input, n_hidden[i]).float()

        # Create the last layer of NN
        self.layers[str(self.depth)] = ConvexLinear(n_hidden[self.depth - 1], n_output).float()
        self.skip_layers[str(self.depth)] = ConvexLinear(n_input, n_output).float()

    def forward(self, F):
        # Get components of deformation gradient tensor F
        # [:,n:n+1] here preserves the original dimensions of x
        F11 = F[:, 0:1]
        F12 = F[:, 1:2]
        F13 = F[:, 2:3]
        F21 = F[:, 3:4]
        F22 = F[:, 4:5]
        F23 = F[:, 5:6]
        F31 = F[:, 6:7]
        F32 = F[:, 7:8]
        F33 = F[:, 8:9]

        # Compute right Cauchy-Green strain tensor
        C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
        C12 = F11 * F12 + F21 * F22 + F31 * F32
        C13 = F11 * F13 + F21 * F23 + F31 * F33
        C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
        C23 = F12 * F13 + F22 * F23 + F32 * F33
        C33 = F13 ** 2 + F23 ** 2 + F33 ** 2

        # Compute strain invariants
        I1 = C11 + C22 + C33
        I2 = 0.5 * (I1 ** 2 - C11 ** 2 - C22 ** 2 - C33 ** 2) - C12 ** 2 - C13 ** 2 - C23 ** 2
        I3 = C11 * C22 * C33 + 2 * C12 * C23 * C13 - C11 * C23 ** 2 - C22 * C13 ** 2 - C33 * C12 ** 2

        # Apply transformation to invariants
        K1 = I1 * torch.pow(I3, -1.0 / 3.0) - 3.0
        K2 = I2 * torch.pow(I3, -2.0 / 3.0) - 3.0
        K3 = (torch.sqrt(I3) - 1) ** 2

        if self.bypass:
            W = 2 * K1 + 1 * K3
            return W

        # Concatenate feature
        x_input = torch.cat((K1, K2, K3), dim=1).float()

        # Forward propagation
        z = x_input.clone()
        z = self.layers[str(0)](z)
        for layer in range(1, self.depth):
            skip = self.skip_layers[str(layer)](x_input)
            z = self.layers[str(layer)](z)
            z += skip
            z = torch.nn.functional.softplus(z)
            if use_sftpSquared:
                z = scaling_sftpSq * torch.square(z)
            if self.training:
                if self.dropout:
                    z = torch.nn.functional.dropout(z, p=self.p_dropout)
        y = self.layers[str(self.depth)](z) + self.skip_layers[str(self.depth)](x_input)
        return y


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        # Xavier uniform initialization
        # maintain the distribution of activation values
        torch.nn.init.xavier_uniform_(m.weight)


def print_nn_architecture(model):
    print('-' * num_marker)
    print("Model architecture:")
    print(model)
    print('-' * num_marker)


def print_nn_params(model):
    print('-' * num_marker)
    print("Initial model:")
    print(model.state_dict())
    print('-' * num_marker)
