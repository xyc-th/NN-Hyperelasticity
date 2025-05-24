from .config import *


def compute_value(data, model, type_name=None):
    if type_name is None:
        raise ValueError("Specify the type of value to be calculated: 'pk_stress', 'nodal_force', or 'loss'.")
    if type_name != 'energy' and type_name != 'pk_stress' and type_name != 'nodal_force' and type_name != 'loss' and type_name != 'pk_stress & energy':
        raise ValueError("Wrong value type. Choose among 'pk_stress', 'nodal_force', 'loss' and 'pk_stress & energy'.")

    # Get components of F from dataset
    F11 = data.F[:, 0:1]
    F12 = data.F[:, 1:2]
    F13 = data.F[:, 2:3]
    F21 = data.F[:, 3:4]
    F22 = data.F[:, 4:5]
    F23 = data.F[:, 5:6]
    F31 = data.F[:, 6:7]
    F32 = data.F[:, 7:8]
    F33 = data.F[:, 8:9]

    # Allow to build computational graph
    F11.requires_grad = True
    F12.requires_grad = True
    F13.requires_grad = True
    F21.requires_grad = True
    F22.requires_grad = True
    F23.requires_grad = True
    F31.requires_grad = True
    F32.requires_grad = True
    F33.requires_grad = True

    # Forward pass NN to obtain W_NN
    W_NN = model(torch.cat((F11, F12, F13, F21, F22, F23, F31, F32, F33), dim=1))

    # Get gradients of W with respect to F
    dW_NNdF11 = torch.autograd.grad(W_NN, F11, torch.ones(F11.shape[0], 1).to(device), create_graph=True)[0]
    dW_NNdF12 = torch.autograd.grad(W_NN, F12, torch.ones(F12.shape[0], 1).to(device), create_graph=True)[0]
    dW_NNdF13 = torch.autograd.grad(W_NN, F13, torch.ones(F13.shape[0], 1).to(device), create_graph=True)[0]
    dW_NNdF21 = torch.autograd.grad(W_NN, F21, torch.ones(F21.shape[0], 1).to(device), create_graph=True)[0]
    dW_NNdF22 = torch.autograd.grad(W_NN, F22, torch.ones(F22.shape[0], 1).to(device), create_graph=True)[0]
    dW_NNdF23 = torch.autograd.grad(W_NN, F23, torch.ones(F23.shape[0], 1).to(device), create_graph=True)[0]
    dW_NNdF31 = torch.autograd.grad(W_NN, F31, torch.ones(F31.shape[0], 1).to(device), create_graph=True)[0]
    dW_NNdF32 = torch.autograd.grad(W_NN, F32, torch.ones(F32.shape[0], 1).to(device), create_graph=True)[0]
    dW_NNdF33 = torch.autograd.grad(W_NN, F33, torch.ones(F33.shape[0], 1).to(device), create_graph=True)[0]

    # Assemble First Piola-Kirchhoff stress components
    P_NN = torch.cat((dW_NNdF11, dW_NNdF12, dW_NNdF13,
                      dW_NNdF21, dW_NNdF22, dW_NNdF23,
                      dW_NNdF31, dW_NNdF32, dW_NNdF33), dim=1)

    # Initialize deformation gradient tensor F_0 = I
    # F_0 represents zero deformation dummy data
    F_0 = torch.zeros((1, 9)).to(device)
    F_0[:, 0] = 1
    F_0[:, 4] = 1
    F_0[:, 8] = 1

    # Components of F_0
    F11_0 = F_0[:, 0:1]
    F12_0 = F_0[:, 1:2]
    F13_0 = F_0[:, 2:3]
    F21_0 = F_0[:, 3:4]
    F22_0 = F_0[:, 4:5]
    F23_0 = F_0[:, 5:6]
    F31_0 = F_0[:, 6:7]
    F32_0 = F_0[:, 7:8]
    F33_0 = F_0[:, 8:9]

    # Require gradient
    F11_0.requires_grad = True
    F12_0.requires_grad = True
    F13_0.requires_grad = True
    F21_0.requires_grad = True
    F22_0.requires_grad = True
    F23_0.requires_grad = True
    F31_0.requires_grad = True
    F32_0.requires_grad = True
    F33_0.requires_grad = True

    # Forward propagation F_0 to obtain zero-deformation energy corrections
    W_NN_0 = model(torch.cat((F11_0, F12_0, F13_0,
                              F21_0, F22_0, F23_0,
                              F31_0, F32_0, F33_0), dim=1))

    if type_name == 'energy':
        return W_NN - W_NN_0

    # Get gradients of W_NN_0 with respect to F_0
    dW_NN_0dF11_0 = torch.autograd.grad(W_NN_0, F11_0, torch.ones(F11_0.shape[0], 1).to(device), create_graph=True)[0]
    dW_NN_0dF12_0 = torch.autograd.grad(W_NN_0, F12_0, torch.ones(F12_0.shape[0], 1).to(device), create_graph=True)[0]
    dW_NN_0dF13_0 = torch.autograd.grad(W_NN_0, F13_0, torch.ones(F13_0.shape[0], 1).to(device), create_graph=True)[0]
    dW_NN_0dF21_0 = torch.autograd.grad(W_NN_0, F21_0, torch.ones(F21_0.shape[0], 1).to(device), create_graph=True)[0]
    dW_NN_0dF22_0 = torch.autograd.grad(W_NN_0, F22_0, torch.ones(F22_0.shape[0], 1).to(device), create_graph=True)[0]
    dW_NN_0dF23_0 = torch.autograd.grad(W_NN_0, F23_0, torch.ones(F23_0.shape[0], 1).to(device), create_graph=True)[0]
    dW_NN_0dF31_0 = torch.autograd.grad(W_NN_0, F31_0, torch.ones(F31_0.shape[0], 1).to(device), create_graph=True)[0]
    dW_NN_0dF32_0 = torch.autograd.grad(W_NN_0, F32_0, torch.ones(F32_0.shape[0], 1).to(device), create_graph=True)[0]
    dW_NN_0dF33_0 = torch.autograd.grad(W_NN_0, F33_0, torch.ones(F33_0.shape[0], 1).to(device), create_graph=True)[0]

    # Get First Piola-Kirchhoff stress at zero deformation
    P_NN_0 = torch.cat((dW_NN_0dF11_0, dW_NN_0dF12_0, dW_NN_0dF13_0,
                        dW_NN_0dF21_0, dW_NN_0dF22_0, dW_NN_0dF23_0,
                        dW_NN_0dF31_0, dW_NN_0dF32_0, dW_NN_0dF33_0), dim=1)

    # Initialize zero stress correction term
    P_correct = torch.zeros_like(P_NN).to(device)

    # Compute components of zero stress correction term according to Ansatz
    P_correct[:, 0:1] = -(F11 * P_NN_0[:, 0:1] + F12 * P_NN_0[:, 3:4] + F13 * P_NN_0[:, 6:7])
    P_correct[:, 1:2] = -(F11 * P_NN_0[:, 1:2] + F12 * P_NN_0[:, 4:5] + F13 * P_NN_0[:, 7:8])
    P_correct[:, 2:3] = -(F11 * P_NN_0[:, 2:3] + F12 * P_NN_0[:, 5:6] + F13 * P_NN_0[:, 8:9])
    P_correct[:, 3:4] = -(F21 * P_NN_0[:, 0:1] + F22 * P_NN_0[:, 3:4] + F23 * P_NN_0[:, 6:7])
    P_correct[:, 4:5] = -(F21 * P_NN_0[:, 1:2] + F22 * P_NN_0[:, 4:5] + F23 * P_NN_0[:, 7:8])
    P_correct[:, 5:6] = -(F21 * P_NN_0[:, 2:3] + F22 * P_NN_0[:, 5:6] + F23 * P_NN_0[:, 8:9])
    P_correct[:, 6:7] = -(F31 * P_NN_0[:, 0:1] + F32 * P_NN_0[:, 3:4] + F33 * P_NN_0[:, 6:7])
    P_correct[:, 7:8] = -(F31 * P_NN_0[:, 1:2] + F32 * P_NN_0[:, 4:5] + F33 * P_NN_0[:, 7:8])
    P_correct[:, 8:9] = -(F31 * P_NN_0[:, 2:3] + F32 * P_NN_0[:, 5:6] + F33 * P_NN_0[:, 8:9])

    # Compute final stress (NN + correction term)
    P = P_NN + P_correct

    if type_name == 'pk_stress':
        return P

    if type_name == 'pk_stress & energy':
        return P, W_NN - W_NN_0

    # Compute nodal forces
    # Number of nodal force components equals to total degree of freedom
    rf_node = torch.zeros(data.num_node, dim).to(device)
    for a in range(num_nodes_per_element):  # Iterate over each node of one element
        for i in range(dim):  # Iterate over each direction of the nodal force (xyz)
            for j in range(dim):  # Iterate over the components of the Voigt matrix
                # Calculate the contribution of the current integrate point to the nodal force f(a,i)
                f_contribution = P[:, voigt_map[i][j]] * data.grad_shape_func[:, a, j] * data.volume_element
                # Accumulate the individual f(a,i) to the corresponding position of the global nodal force F(n,i)
                rf_node[:, i].index_add_(0, data.connectivity[:, a], f_contribution)

    if type_name == 'nodal_force':
        return rf_node

    # Compute loss based on PK stress
    stress_loss = torch.sum((P - data.P) ** 2)

    # Loss for force equilibrium
    eq_loss = torch.sum(rf_node[~data.dirichlet_node] ** 2)  # internal loss
    bc_loss = torch.zeros(1, dtype=torch.float).to(device)
    for bc_rank in range(data.num_bc):
        bc_loss += torch.sum(
            (torch.sum(rf_node[data.bc_node[:, bc_rank] == 1], dim=0) - data.rf_global[bc_rank]) ** 2)  # boundary loss

    if type_name == 'loss':
        return eq_loss, bc_loss, stress_loss


def train_model(model, datasets):
    # Learning rate
    lr_history = []

    # Loss history
    loss_history = []
    eq_loss_history = []
    bc_loss_history = []
    stress_loss_history = []

    # Optimizer
    if optimization_method == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimization_method == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
    elif optimization_method == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimization_method == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Incorrect choice of optimizer. Choose among 'adam', 'lbfgs', 'sgd', and 'rmsprop'.")

    # Scheduler
    if lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=le_milestones, gamma=lr_decay,
                                                         last_epoch=-1)
    elif lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                                                      cycle_momentum=cycle_momentum,
                                                      step_size_up=step_size_up, step_size_down=step_size_down)
    elif lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif lr_schedule == 'cosine_warm':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult,
                                                                         eta_min=eta_min)
    elif lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    else:
        raise ValueError("Incorrect choice of scheduler. Choose between 'multistep' and 'cyclic'.")

    # Training of NN
    print('-' * num_marker)
    print("|   epoch   |    lr    |     loss     |      eq      |      bc      |     stress    |")
    print('-' * num_marker)
    for epoch_iter in range(epochs):
        lr_history.append(scheduler.get_last_lr()[0])
        optimizer.zero_grad()

        # Compute loss for each displacement snapshot in dataset
        loss = torch.zeros(1, dtype=torch.float).to(device)
        eq_loss = torch.zeros(1, dtype=torch.float).to(device)
        bc_loss = torch.zeros(1, dtype=torch.float).to(device)
        stress_loss = torch.zeros(1, dtype=torch.float).to(device)
        for data in datasets:
            eq_loss_data, bc_loss_data, stress_loss_data = compute_value(data, model, type_name='loss')
            loss += eqb_loss_factor * eq_loss_data + bc_loss_factor * bc_loss_data
            eq_loss += eqb_loss_factor * eq_loss_data
            bc_loss += bc_loss_factor * bc_loss_data
            stress_loss += stress_loss_data

        # Backward propagation
        loss.backward()

        # Step for optimizer and scheduler
        optimizer.step()
        scheduler.step(loss)

        # Real-time output of training process
        print(
            f"| {epoch_iter + 1:4}/{epochs:<4} | {optimizer.param_groups[0]['lr']:<8.2e} "
            f"| {loss.item():<12.6e} | {eq_loss.item():<12.6e} | {bc_loss.item():<12.6e} "
            f"| {stress_loss.item():<13.6e} |")

        # Real-time saving for loss values
        loss_history.append(loss.item())
        eq_loss_history.append(eq_loss.item())
        bc_loss_history.append(bc_loss.item())
        stress_loss_history.append(stress_loss.item())

    print('-' * num_marker)
    return model, loss_history, eq_loss_history, bc_loss_history, stress_loss_history, lr_history
