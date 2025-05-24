from .config import *


def convert_tensor_to_numpy(T):
    """
    Convert T to numpy array.
    :param T:
    :return:
    """
    T = T.cpu().detach().numpy()
    return T


def compute_jacobian(F):
    """
    Compute Jacobian J from deformation gradient tensor F.
    :param F: deformation gradient F in Voigt notation
    :return J: Jacobian of F
    """
    F11 = F[:, 0:1]
    F12 = F[:, 1:2]
    F13 = F[:, 2:3]
    F21 = F[:, 3:4]
    F22 = F[:, 4:5]
    F23 = F[:, 5:6]
    F31 = F[:, 6:7]
    F32 = F[:, 7:8]
    F33 = F[:, 8:9]

    J = F11 * F22 * F33 + F12 * F23 * F31 + F13 * F32 * F21 - F11 * F23 * F32 - F22 * F13 * F31 - F33 * F12 * F21
    return J


def compute_cauchy_green_strain(F):
    """
    Compute right Cauchy-Green strain tensor C from deformation gradient tensor F.
    :param F: deformation gradient in Voigt notation
    :return C: right Cauchy-Green strain tensor in Voigt notation
    """
    F11 = F[:, 0:1]
    F12 = F[:, 1:2]
    F13 = F[:, 2:3]
    F21 = F[:, 3:4]
    F22 = F[:, 4:5]
    F23 = F[:, 5:6]
    F31 = F[:, 6:7]
    F32 = F[:, 7:8]
    F33 = F[:, 8:9]

    C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
    C12 = F11 * F12 + F21 * F22 + F31 * F32
    C13 = F11 * F13 + F21 * F23 + F31 * F33
    C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
    C23 = F12 * F13 + F22 * F23 + F32 * F33
    C33 = F13 ** 2 + F23 ** 2 + F33 ** 2

    C = torch.cat((C11, C12, C13, C12, C22, C23, C13, C23, C33), dim=1)
    return C


def compute_strain_invariants(C):
    """
    Compute invariants of the right Cauchy-Green strain tensor C
    :param C: right Cauchy-Green strain tensor in Voigt notation
    :return I1: 1st invariant
    :return I2: 2nd invariant
    :return I3: 3rd invariant
    """
    C11 = C[:, 0:1]
    C12 = C[:, 1:2]
    C13 = C[:, 2:3]
    C22 = C[:, 4:5]
    C23 = C[:, 5:6]
    C33 = C[:, 8:9]

    I1 = C11 + C22 + C33
    I2 = 0.5 * (I1 ** 2 - C11 ** 2 - C22 ** 2 - C33 ** 2) - C12 ** 2 - C13 ** 2 - C23 ** 2
    I3 = C11 * C22 * C33 + 2 * C12 * C23 * C13 - C11 * C23 ** 2 - C22 * C13 ** 2 - C33 * C12 ** 2
    return I1, I2, I3


def compute_strain_invariant_derivatives(F, i, second_derivative=False):
    """
    Compute derivatives of the invariants of the right Cauchy-Green strain tensor
    with respect to the deformation gradient tensor F.
    :param F: deformation gradient tensor in Voigt notation
    :param i: specify the invariant that should be differentiated
    :param second_derivative: specify if second derivative should be computed
    :return dIdF: derivative, note that the size of 'dIdF' depends on the choice of 'second_derivative'
    """
    F11 = F[:, 0:1]
    F12 = F[:, 1:2]
    F13 = F[:, 2:3]
    F21 = F[:, 3:4]
    F22 = F[:, 4:5]
    F23 = F[:, 5:6]
    F31 = F[:, 6:7]
    F32 = F[:, 7:8]
    F33 = F[:, 8:9]

    if not second_derivative:
        # dIdF = torch.zeros(F.shape[0], F.shape[1])
        if i == 1:  # dI1/dF
            dIdF = 2.0 * F

        elif i == 2:  # dI2/dF
            C = compute_cauchy_green_strain(F)
            C11 = C[:, 0:1]
            C12 = C[:, 1:2]
            C13 = C[:, 2:3]
            C22 = C[:, 4:5]
            C23 = C[:, 5:6]
            C33 = C[:, 8:9]
            I1 = C11 + C22 + C33

            dIdF11 = 2.0 * (I1 * F11 - (F11 * C11 + F12 * C12 + F13 * C13))
            dIdF12 = 2.0 * (I1 * F12 - (F11 * C12 + F12 * C22 + F13 * C23))
            dIdF13 = 2.0 * (I1 * F13 - (F11 * C13 + F12 * C23 + F13 * C33))
            dIdF21 = 2.0 * (I1 * F21 - (F21 * C11 + F22 * C12 + F23 * C13))
            dIdF22 = 2.0 * (I1 * F22 - (F21 * C12 + F22 * C22 + F23 * C23))
            dIdF23 = 2.0 * (I1 * F23 - (F21 * C13 + F22 * C23 + F23 * C33))
            dIdF31 = 2.0 * (I1 * F31 - (F31 * C11 + F32 * C12 + F33 * C13))
            dIdF32 = 2.0 * (I1 * F32 - (F31 * C12 + F32 * C22 + F33 * C23))
            dIdF33 = 2.0 * (I1 * F33 - (F31 * C13 + F32 * C23 + F33 * C33))
            dIdF = torch.cat((dIdF11, dIdF12, dIdF13, dIdF21, dIdF22, dIdF23, dIdF31, dIdF32, dIdF33), dim=1)

        elif i == 3:  # dI3/dF
            J = compute_jacobian(F)

            dIdF11 = 2.0 * J * (F22 * F33 - F23 * F32)
            dIdF12 = 2.0 * J * (F23 * F31 - F21 * F33)
            dIdF13 = 2.0 * J * (F21 * F32 - F22 * F31)
            dIdF21 = 2.0 * J * (F13 * F32 - F12 * F33)
            dIdF22 = 2.0 * J * (F11 * F33 - F13 * F31)
            dIdF23 = 2.0 * J * (F12 * F31 - F11 * F32)
            dIdF31 = 2.0 * J * (F12 * F23 - F22 * F13)
            dIdF32 = 2.0 * J * (F13 * F21 - F11 * F23)
            dIdF33 = 2.0 * J * (F11 * F22 - F12 * F21)
            dIdF = torch.cat((dIdF11, dIdF12, dIdF13, dIdF21, dIdF22, dIdF23, dIdF31, dIdF32, dIdF33), dim=1)

        else:
            raise ValueError('Incorrect invariant index.')
    else:
        dIdF = torch.zeros(F.shape[1], F.shape[1])
        if i == 1:  # d(dI1/dF)/dF
            dIdF = 2.0 * torch.eye(F.shape[1])

        elif i == 3:  # d(dI3/dF)/dF
            J = compute_jacobian(F)
            dJdF11 = F22 * F33 - F23 * F32
            dJdF12 = F23 * F31 - F21 * F33
            dJdF13 = F21 * F32 - F22 * F31
            dJdF21 = F13 * F32 - F12 * F33
            dJdF22 = F11 * F33 - F13 * F31
            dJdF23 = F12 * F31 - F11 * F32
            dJdF31 = F12 * F23 - F22 * F13
            dJdF32 = F13 * F21 - F11 * F23
            dJdF33 = F11 * F22 - F12 * F21

            # d(dI3/dF)/dF11
            dIdF[0, 0] = 2.0 * dJdF11 * dJdF11
            dIdF[0, 1] = 2.0 * dJdF11 * dJdF12
            dIdF[0, 2] = 2.0 * dJdF11 * dJdF13
            dIdF[0, 3] = 2.0 * dJdF11 * dJdF21
            dIdF[0, 4] = 2.0 * dJdF11 * dJdF22 + 2.0 * J * F33
            dIdF[0, 5] = 2.0 * dJdF11 * dJdF23 - 2.0 * J * F32
            dIdF[0, 6] = 2.0 * dJdF11 * dJdF31
            dIdF[0, 7] = 2.0 * dJdF11 * dJdF32 - 2.0 * J * F23
            dIdF[0, 8] = 2.0 * dJdF11 * dJdF33 + 2.0 * J * F22

            # d(dI3/dF)/dF12
            dIdF[1, 0] = 2.0 * dJdF12 * dJdF11
            dIdF[1, 1] = 2.0 * dJdF12 * dJdF12
            dIdF[1, 2] = 2.0 * dJdF12 * dJdF13
            dIdF[1, 3] = 2.0 * dJdF12 * dJdF21 - 2.0 * J * F33
            dIdF[1, 4] = 2.0 * dJdF12 * dJdF22
            dIdF[1, 5] = 2.0 * dJdF12 * dJdF23 + 2.0 * J * F31
            dIdF[1, 6] = 2.0 * dJdF12 * dJdF31 + 2.0 * J * F23
            dIdF[1, 7] = 2.0 * dJdF12 * dJdF32
            dIdF[1, 8] = 2.0 * dJdF12 * dJdF33 - 2.0 * J * F21

            # d(dI3/dF)/dF13
            dIdF[2, 0] = 2.0 * dJdF13 * dJdF11
            dIdF[2, 1] = 2.0 * dJdF13 * dJdF12
            dIdF[2, 2] = 2.0 * dJdF13 * dJdF13
            dIdF[2, 3] = 2.0 * dJdF13 * dJdF21 + 2.0 * J * F32
            dIdF[2, 4] = 2.0 * dJdF13 * dJdF22 - 2.0 * J * F31
            dIdF[2, 5] = 2.0 * dJdF13 * dJdF23
            dIdF[2, 6] = 2.0 * dJdF13 * dJdF31 - 2.0 * J * F22
            dIdF[2, 7] = 2.0 * dJdF13 * dJdF32 + 2.0 * J * F21
            dIdF[2, 8] = 2.0 * dJdF13 * dJdF33

            # d(dI3/dF)/dF21
            dIdF[3, 0] = 2.0 * dJdF21 * dJdF11
            dIdF[3, 1] = 2.0 * dJdF21 * dJdF12 - 2.0 * J * F33
            dIdF[3, 2] = 2.0 * dJdF21 * dJdF13 + 2.0 * J * F32
            dIdF[3, 3] = 2.0 * dJdF21 * dJdF21
            dIdF[3, 4] = 2.0 * dJdF21 * dJdF22
            dIdF[3, 5] = 2.0 * dJdF21 * dJdF23
            dIdF[3, 6] = 2.0 * dJdF21 * dJdF31
            dIdF[3, 7] = 2.0 * dJdF21 * dJdF32 + 2.0 * J * F13
            dIdF[3, 8] = 2.0 * dJdF21 * dJdF33 - 2.0 * J * F12

            # d(dI3/dF)/dF22
            dIdF[4, 0] = 2.0 * dJdF22 * dJdF11 + 2.0 * J * F33
            dIdF[4, 1] = 2.0 * dJdF22 * dJdF12
            dIdF[4, 2] = 2.0 * dJdF22 * dJdF13 - 2.0 * J * F31
            dIdF[4, 3] = 2.0 * dJdF22 * dJdF21
            dIdF[4, 4] = 2.0 * dJdF22 * dJdF22
            dIdF[4, 5] = 2.0 * dJdF22 * dJdF23
            dIdF[4, 6] = 2.0 * dJdF22 * dJdF31 - 2.0 * J * F13
            dIdF[4, 7] = 2.0 * dJdF22 * dJdF32
            dIdF[4, 8] = 2.0 * dJdF22 * dJdF33 + 2.0 * J * F11

            # d(dI3/dF)/dF23
            dIdF[5, 0] = 2.0 * dJdF23 * dJdF11 - 2.0 * J * F32
            dIdF[5, 1] = 2.0 * dJdF23 * dJdF12 + 2.0 * J * F31
            dIdF[5, 2] = 2.0 * dJdF23 * dJdF13
            dIdF[5, 3] = 2.0 * dJdF23 * dJdF21
            dIdF[5, 4] = 2.0 * dJdF23 * dJdF22
            dIdF[5, 5] = 2.0 * dJdF23 * dJdF23
            dIdF[5, 6] = 2.0 * dJdF23 * dJdF31 + 2.0 * J * F12
            dIdF[5, 7] = 2.0 * dJdF23 * dJdF32 - 2.0 * J * F11
            dIdF[5, 8] = 2.0 * dJdF23 * dJdF33

            # d(dI3/dF)/dF31
            dIdF[6, 0] = 2.0 * dJdF31 * dJdF11
            dIdF[6, 1] = 2.0 * dJdF31 * dJdF12 + 2.0 * J * F23
            dIdF[6, 2] = 2.0 * dJdF31 * dJdF13 - 2.0 * J * F22
            dIdF[6, 3] = 2.0 * dJdF31 * dJdF21
            dIdF[6, 4] = 2.0 * dJdF31 * dJdF22 - 2.0 * J * F13
            dIdF[6, 5] = 2.0 * dJdF31 * dJdF23 + 2.0 * J * F12
            dIdF[6, 6] = 2.0 * dJdF31 * dJdF31
            dIdF[6, 7] = 2.0 * dJdF31 * dJdF32
            dIdF[6, 8] = 2.0 * dJdF31 * dJdF33

            # d(dI3/dF)/dF32
            dIdF[7, 0] = 2.0 * dJdF32 * dJdF11 - 2.0 * J * F23
            dIdF[7, 1] = 2.0 * dJdF32 * dJdF12
            dIdF[7, 2] = 2.0 * dJdF32 * dJdF13 + 2.0 * J * F21
            dIdF[7, 3] = 2.0 * dJdF32 * dJdF21 + 2.0 * J * F13
            dIdF[7, 4] = 2.0 * dJdF32 * dJdF22
            dIdF[7, 5] = 2.0 * dJdF32 * dJdF23 - 2.0 * J * F11
            dIdF[7, 6] = 2.0 * dJdF32 * dJdF31
            dIdF[7, 7] = 2.0 * dJdF32 * dJdF32
            dIdF[7, 8] = 2.0 * dJdF32 * dJdF33

            # d(dI3/dF)/dF33
            dIdF[8, 0] = 2.0 * dJdF33 * dJdF11 + 2.0 * J * F22
            dIdF[8, 1] = 2.0 * dJdF33 * dJdF12 - 2.0 * J * F21
            dIdF[8, 2] = 2.0 * dJdF33 * dJdF13
            dIdF[8, 3] = 2.0 * dJdF33 * dJdF21 - 2.0 * J * F12
            dIdF[8, 4] = 2.0 * dJdF33 * dJdF22 + 2.0 * J * F11
            dIdF[8, 5] = 2.0 * dJdF33 * dJdF23
            dIdF[8, 6] = 2.0 * dJdF33 * dJdF31
            dIdF[8, 7] = 2.0 * dJdF33 * dJdF32
            dIdF[8, 8] = 2.0 * dJdF33 * dJdF33

        else:
            raise ValueError("Incorrect invariant index.")

    return dIdF


def compute_features(I1, I2, I3):
    """
    Compute features dependent on the right Cauchy-Green strain invariants.
    :param I1: 1st invariant, I1 = tr(C)
    :param I2: 2nd invariant, I2 = 1/2*(tr^2(C)-tr(C^2))
    :param I3: 3rd invariant, I3 = det(C)
    :return x: features
    """

    # Compute normalized invariants
    K1 = I1 * torch.pow(I3, -1.0 / 3.0) - 3.0
    K2 = I2 * torch.pow(I3, -2.0 / 3.0) - 3.0
    K3 = (torch.sqrt(I3) - 1) ** 2

    x = torch.cat((K1, K2, K3), dim=1).float()
    return x


def pk_to_cauchy(F, P, original=False):
    J = compute_jacobian(F)
    sigma_original = torch.zeros_like(P)

    sigma_original[:, 0] = P[:, 0] * F[:, 0] + P[:, 1] * F[:, 1] + P[:, 2] * F[:, 2]
    sigma_original[:, 1] = P[:, 0] * F[:, 3] + P[:, 1] * F[:, 4] + P[:, 2] * F[:, 5]
    sigma_original[:, 2] = P[:, 0] * F[:, 6] + P[:, 1] * F[:, 7] + P[:, 2] * F[:, 8]
    sigma_original[:, 3] = P[:, 3] * F[:, 0] + P[:, 4] * F[:, 1] + P[:, 5] * F[:, 2]
    sigma_original[:, 4] = P[:, 3] * F[:, 3] + P[:, 4] * F[:, 4] + P[:, 5] * F[:, 5]
    sigma_original[:, 5] = P[:, 3] * F[:, 6] + P[:, 4] * F[:, 7] + P[:, 5] * F[:, 8]
    sigma_original[:, 6] = P[:, 6] * F[:, 0] + P[:, 7] * F[:, 1] + P[:, 8] * F[:, 2]
    sigma_original[:, 7] = P[:, 6] * F[:, 3] + P[:, 7] * F[:, 4] + P[:, 8] * F[:, 5]
    sigma_original[:, 8] = P[:, 6] * F[:, 6] + P[:, 7] * F[:, 7] + P[:, 8] * F[:, 8]

    sigma_original = sigma_original / J
    if original:
        return sigma_original
    else:
        sigma = torch.zeros((sigma_original.shape[0], 6))
        sigma[:, 0] = sigma_original[:, 0]
        sigma[:, 1] = sigma_original[:, 4]
        sigma[:, 2] = sigma_original[:, 8]
        sigma[:, 3] = 0.5 * (sigma_original[:, 1] + sigma_original[:, 3])
        sigma[:, 4] = 0.5 * (sigma_original[:, 2] + sigma_original[:, 6])
        sigma[:, 5] = 0.5 * (sigma_original[:, 5] + sigma_original[:, 7])
        return sigma


def cauchy_to_pk(F, sigma):
    P = torch.zeros(sigma.shape[0], 9)
    F_adj = torch.zeros_like(F)

    # Adjugate matrix of F, adj(F)
    F_adj[:, 0] = F[:, 4] * F[:, 8] - F[:, 5] * F[:, 7]
    F_adj[:, 1] = F[:, 5] * F[:, 6] - F[:, 3] * F[:, 8]
    F_adj[:, 2] = F[:, 3] * F[:, 7] - F[:, 4] * F[:, 6]
    F_adj[:, 3] = F[:, 2] * F[:, 7] - F[:, 1] * F[:, 8]
    F_adj[:, 4] = F[:, 0] * F[:, 8] - F[:, 2] * F[:, 6]
    F_adj[:, 5] = F[:, 1] * F[:, 6] - F[:, 0] * F[:, 7]
    F_adj[:, 6] = F[:, 1] * F[:, 5] - F[:, 2] * F[:, 4]
    F_adj[:, 7] = F[:, 2] * F[:, 3] - F[:, 0] * F[:, 5]
    F_adj[:, 8] = F[:, 0] * F[:, 4] - F[:, 1] * F[:, 3]

    P[:, 0] = sigma[:, 0] * F_adj[:, 0] + sigma[:, 3] * F_adj[:, 3] + sigma[:, 4] * F_adj[:, 6]
    P[:, 1] = sigma[:, 0] * F_adj[:, 1] + sigma[:, 3] * F_adj[:, 4] + sigma[:, 4] * F_adj[:, 7]
    P[:, 2] = sigma[:, 0] * F_adj[:, 2] + sigma[:, 3] * F_adj[:, 5] + sigma[:, 4] * F_adj[:, 8]
    P[:, 3] = sigma[:, 3] * F_adj[:, 0] + sigma[:, 1] * F_adj[:, 3] + sigma[:, 5] * F_adj[:, 6]
    P[:, 4] = sigma[:, 3] * F_adj[:, 1] + sigma[:, 1] * F_adj[:, 4] + sigma[:, 5] * F_adj[:, 7]
    P[:, 5] = sigma[:, 3] * F_adj[:, 2] + sigma[:, 1] * F_adj[:, 5] + sigma[:, 5] * F_adj[:, 8]
    P[:, 6] = sigma[:, 4] * F_adj[:, 0] + sigma[:, 5] * F_adj[:, 3] + sigma[:, 2] * F_adj[:, 6]
    P[:, 7] = sigma[:, 4] * F_adj[:, 1] + sigma[:, 5] * F_adj[:, 4] + sigma[:, 2] * F_adj[:, 7]
    P[:, 8] = sigma[:, 4] * F_adj[:, 2] + sigma[:, 5] * F_adj[:, 5] + sigma[:, 2] * F_adj[:, 8]

    return P


def cauchy_to_mises(sigma):
    sxx = sigma[:, 0]
    syy = sigma[:, 1]
    szz = sigma[:, 2]
    sxy = sigma[:, 3]
    sxz = sigma[:, 4]
    syz = sigma[:, 5]

    s_mises = torch.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) +
                         3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2))
    return s_mises


def cauchy_error(sigma_model, sigma_real):
    sxx_model = sigma_model[:, 0]
    syy_model = sigma_model[:, 1]
    szz_model = sigma_model[:, 2]
    sxy_model = sigma_model[:, 3]
    sxz_model = sigma_model[:, 4]
    syz_model = sigma_model[:, 5]
    s_mises_model = cauchy_to_mises(sigma_model)

    sxx_real = sigma_real[:, 0]
    syy_real = sigma_real[:, 1]
    szz_real = sigma_real[:, 2]
    sxy_real = sigma_real[:, 3]
    sxz_real = sigma_real[:, 4]
    syz_real = sigma_real[:, 5]
    s_mises_real = cauchy_to_mises(sigma_real)

    sxx_error = torch.norm(sxx_model - sxx_real) / torch.norm(sxx_real)
    syy_error = torch.norm(syy_model - syy_real) / torch.norm(syy_real)
    szz_error = torch.norm(szz_model - szz_real) / torch.norm(szz_real)
    sxy_error = torch.norm(sxy_model - sxy_real) / torch.norm(sxy_real)
    sxz_error = torch.norm(sxz_model - sxz_real) / torch.norm(sxz_real)
    syz_error = torch.norm(syz_model - syz_real) / torch.norm(syz_real)
    s_mises_error = torch.norm(s_mises_model - s_mises_real) / torch.norm(s_mises_real)

    return sxx_error, syy_error, szz_error, sxy_error, sxz_error, syz_error, s_mises_error
