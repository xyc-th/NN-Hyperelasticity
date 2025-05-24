from .gen_util import *


class Dataset:
    def __init__(self, path,
                 x_node, u_node, bc_node, test_bc_node, dirichlet_node, test_bc_all_node,
                 rf_global, connectivity, grad_shape_func, volume_element,
                 F, J, C, I1, I2, I3, S, P):
        """
        Generate 'Dataset' object.
        :param path:
        :param x_node:
        :param u_node:
        :param bc_node:
        :param test_bc_node:
        :param dirichlet_node:
        :param rf_global:
        :param connectivity:
        :param grad_shape_func:
        :param volume_element:
        :param F:
        :param J:
        :param C:
        :param S:
        :param P:
        """
        self.path = path

        # Nodal data
        self.num_node = x_node.shape[0]
        self.x_node = x_node  # Node positions: num_node * 2
        self.u_node = u_node  # Node displacements: num_node * 2
        # Dirichlet_nodes is a boolean matrix of size: num_node * 2
        # Dirichlet_nodes[a,i]=True means node 'a' is constrained in 'i'th direction
        self.num_bc = bc_node.shape[1]
        self.num_test_bc = test_bc_node.shape[1]
        self.bc_node = bc_node
        self.test_bc_node = test_bc_node
        self.dirichlet_node = dirichlet_node  # boolean: num_node * 2
        self.test_bc_all_node = test_bc_all_node  # boolean: num_node * 2

        # Global reaction force
        self.rf_global = rf_global

        # Element data
        self.num_element = volume_element.shape[0]
        # volume_element is the volume of each element
        self.volume_element = volume_element
        # Connectivity is a list of 1D matrix. Each matrix contains 4 integers
        # Length of each matrix is equal to num_element
        # For 3D tetrahedral element, a={0,1,2,3}
        # Connectivity[a][e] is the index of the 'a'th node of the 'e'th element
        # Node range from 0 to (num_node - 1)
        self.connectivity = connectivity
        # grad_shape_func is a list of 3 matrices, each of size is num_element * 2
        # For 3D tetrahedral element, a={0,1,2,3} and i={0,1,2}
        # grad_shape_func[a][e,i] is the 'i'th components
        # of the shape function gradient N[e,a] of the 'a'th node in the 'e'th element
        # Evaluated at element e's single quadrature point
        self.grad_shape_func = grad_shape_func

        # strain data
        # 2D matrix with number of rows = num_element
        # Each matrix represents certain strain-related quantity
        # Evaluated at the respective element's quadrature points
        self.F = F  # deformation gradient: num_element * 9
        self.J = J  # determinant of F: num_element * 1
        self.C = C  # right Cauchy-Green strain tensor: num_element * 9
        self.I1 = I1  # 1st invariant of C: num_element * 1
        self.I2 = I2  # 2nd invariant of C: num_element * 1
        self.I3 = I3  # 3rd invariant of C: num_element * 1

        # Stress data
        self.S = S  # Element-wise Cauchy stress
        self.P = P  # Element-wise first Piola-Kirchhoff stress

    def convert_to_numpy(self):
        """
        Convert to numpy array.
        :return:
        """
        # Nodal data
        self.x_node = convert_tensor_to_numpy(self.x_node)
        self.u_node = convert_tensor_to_numpy(self.u_node)
        self.dirichlet_node = convert_tensor_to_numpy(self.dirichlet_node)

        # Reaction
        for i in range(len(self.rf_global)):
            self.rf_global[i].dofs = convert_tensor_to_numpy(self.rf_global[i].dofs)

        # Element data
        for i in range(len(self.connectivity)):
            self.connectivity[i] = convert_tensor_to_numpy(self.connectivity[i])
        for i in range(len(self.grad_shape_func)):
            self.grad_shape_func[i] = convert_tensor_to_numpy(self.grad_shape_func[i])
        self.volume_element = convert_tensor_to_numpy(self.volume_element)

        # Strain data
        self.F = convert_tensor_to_numpy(self.F)
        self.J = convert_tensor_to_numpy(self.J)
        self.C = convert_tensor_to_numpy(self.C)
        self.I1 = convert_tensor_to_numpy(self.I1)
        self.I2 = convert_tensor_to_numpy(self.I2)
        self.I3 = convert_tensor_to_numpy(self.I3)

        # Stress data
        self.S = convert_tensor_to_numpy(self.S)
        self.P = convert_tensor_to_numpy(self.P)


def compute_det(a11, a12, a13, a21, a22, a23, a31, a32, a33):
    return a11 * (a22 * a33 - a23 * a32) - a21 * (a12 * a33 - a13 * a32) + a31 * (a12 * a23 - a13 * a22)


def load_data(path, load_step, noise_type='none', noise_level=0.0):
    """
    Load the input data and add noise (optional).
    Exp data is already perturbed by noise. In this case adding additional noise is not necessary.
    :param path: path to data
    :param load_step: load step of data
    :param noise_level: noise intensity
    :param noise_type: specify whether noise should be added to displacements or strains
    :return dataset: finite element dataset
    """
    print('-' * num_marker)
    print(f"Loading data: {path}/{load_step}")

    # Noise
    if noise_type != 'none' and noise_type != 'disp' and noise_type != 'relative_disp' \
            and noise_type != 'deformation_grad' and noise_type != 'relative_deformation_grad':
        raise ValueError("Incorrect noise type argument! Choose among 'none', 'disp', 'relative_disp',"
                         " 'deformation_grad', or 'relative_deformation_grad'.")

    # Nodal data
    data_node = pd.read_csv(f"{path}/node.csv")
    x_node = torch.tensor(data_node[['x', 'y', 'z']].values, dtype=torch.float).to(device)

    # Boundary data
    data_bc = pd.read_csv(f"{path}/bc.csv")
    bc_node = torch.tensor(data_bc.values, dtype=torch.int).to(device)
    dirichlet_node = torch.any(bc_node == 1, dim=1).to(device)

    # Boundary for test data
    data_test_bc = pd.read_csv(f"{path}/bc_test.csv")
    test_bc_node = torch.tensor(data_test_bc.values, dtype=torch.int).to(device)
    test_bc_all_node = torch.any(test_bc_node == 1, dim=1).to(device)

    # Element data
    data_element = pd.read_csv(f"{path}/element.csv")
    num_element = data_element.shape[0]
    connectivity = torch.tensor(data_element[['a', 'b', 'c', 'd']].values, dtype=torch.int).to(device)

    # Shape function
    x1 = x_node[connectivity[:, 0], 0]
    y1 = x_node[connectivity[:, 0], 1]
    z1 = x_node[connectivity[:, 0], 2]
    x2 = x_node[connectivity[:, 1], 0]
    y2 = x_node[connectivity[:, 1], 1]
    z2 = x_node[connectivity[:, 1], 2]
    x3 = x_node[connectivity[:, 2], 0]
    y3 = x_node[connectivity[:, 2], 1]
    z3 = x_node[connectivity[:, 2], 2]
    x4 = x_node[connectivity[:, 3], 0]
    y4 = x_node[connectivity[:, 3], 1]
    z4 = x_node[connectivity[:, 3], 2]
    volume_element = (+ compute_det(x2, y2, z2, x3, y3, z3, x4, y4, z4)
                      - compute_det(x1, y1, z1, x3, y3, z3, x4, y4, z4)
                      + compute_det(x1, y1, z1, x2, y2, z2, x4, y4, z4)
                      - compute_det(x1, y1, z1, x2, y2, z2, x3, y3, z3))
    grad_shape_func = torch.zeros((num_element, num_nodes_per_element, dim)).to(device)
    grad_shape_func[:, 0, 0] = - compute_det(1, y2, z2, 1, y3, z3, 1, y4, z4) / volume_element
    grad_shape_func[:, 0, 1] = + compute_det(1, x2, z2, 1, x3, z3, 1, x4, z4) / volume_element
    grad_shape_func[:, 0, 2] = - compute_det(1, x2, y2, 1, x3, y3, 1, x4, y4) / volume_element
    grad_shape_func[:, 1, 0] = + compute_det(1, y1, z1, 1, y3, z3, 1, y4, z4) / volume_element
    grad_shape_func[:, 1, 1] = - compute_det(1, x1, z1, 1, x3, z3, 1, x4, z4) / volume_element
    grad_shape_func[:, 1, 2] = + compute_det(1, x1, y1, 1, x3, y3, 1, x4, y4) / volume_element
    grad_shape_func[:, 2, 0] = - compute_det(1, y1, z1, 1, y2, z2, 1, y4, z4) / volume_element
    grad_shape_func[:, 2, 1] = + compute_det(1, x1, z1, 1, x2, z2, 1, x4, z4) / volume_element
    grad_shape_func[:, 2, 2] = - compute_det(1, x1, y1, 1, x2, y2, 1, x4, y4) / volume_element
    grad_shape_func[:, 3, 0] = + compute_det(1, y1, z1, 1, y2, z2, 1, y3, z3) / volume_element
    grad_shape_func[:, 3, 1] = - compute_det(1, x1, z1, 1, x2, z2, 1, x3, z3) / volume_element
    grad_shape_func[:, 3, 2] = + compute_det(1, x1, y1, 1, x2, y2, 1, x3, y3) / volume_element

    # Element volume
    volume_element = volume_element / 6.0

    # Displacement data
    data_disp = pd.read_csv(f"{path}/{load_step}/disp.csv")
    u_node = torch.tensor(data_disp[['ux', 'uy', 'uz']].values, dtype=torch.float).to(device)

    # Apply noise to displacement
    if noise_type == 'disp':
        noise_node = noise_level * torch.randn_like(u_node).to(device)
        noise_node[dirichlet_node] = 0.0
        u_node += noise_node
        print(f"Applying absolute noise to displacements, noise level = {noise_level}")

    if noise_type == 'relative_disp':
        noise_node = noise_level * torch.randn_like(u_node).to(device) * u_node
        noise_node[dirichlet_node] = 0.0
        u_node += noise_node
        print(f"Applying relative noise to displacements, noise level = {noise_level}")

    # Reaction force data
    data_rf = pd.read_csv(f"{path}/{load_step}/global_rf.csv", dtype=np.float64)
    rf_global = torch.tensor(data_rf[['fx', 'fy', 'fz']].values, dtype=torch.float).to(device)

    # Compute deformation gradient at quadrature point
    F = torch.zeros(num_element, 9).to(device)
    for a in range(num_nodes_per_element):
        for i in range(dim):
            for j in range(dim):
                F[:, voigt_map[i][j]] += u_node[connectivity[:, a], i] * grad_shape_func[:, a, j]
    F[:, 0] += 1
    F[:, 4] += 1
    F[:, 8] += 1

    # Apply noise to strain
    if noise_type == 'deformation_grad':
        noise_strain = noise_level * torch.randn_like(F)
        F += noise_strain
        print(f"Applying absolute noise to deformation gradient tensor, noise level = {noise_level}")

    if noise_type == 'relative_deformation_grad':
        noise_strain = noise_level * torch.randn_like(F) * F
        F += noise_strain
        print(f"Applying relative noise to deformation gradient tensor, noise level = {noise_level}")

    # Compute determinant of F
    J = compute_jacobian(F)

    # Compute right Cauchy-Green strain C
    C = compute_cauchy_green_strain(F)

    # Compute strain invariants
    I1, I2, I3 = compute_strain_invariants(C)

    # Activate gradients
    I1.requires_grad = True
    I2.requires_grad = True
    I3.requires_grad = True

    # Cauchy stress data
    data_cauchy = pd.read_csv(f"{path}/{load_step}/stress.csv")
    S = torch.tensor(data_cauchy[['sxx', 'syy', 'szz', 'sxy', 'sxz', 'syz']].values, dtype=torch.float).to(device)
    P = cauchy_to_pk(F, S).to(device)

    dataset = Dataset(path, x_node, u_node, bc_node, test_bc_node, dirichlet_node, test_bc_all_node,
                      rf_global, connectivity, grad_shape_func, volume_element,
                      F, J, C, I1, I2, I3, S, P)

    print('-' * num_marker)
    return dataset
