from .config import *
from .gen_util import *
from .data_util import *
from .io_util import *
from .train_util import compute_value


def ensemble_loss_plot(loss_last_iter, best_loss, best_model):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(np.arange(ensemble_size)[loss_last_iter <= best_loss ** conf_threshold],
               loss_last_iter[loss_last_iter <= best_loss ** conf_threshold],
               label='Accepted')
    ax.scatter(np.arange(ensemble_size)[loss_last_iter > best_loss ** conf_threshold],
               loss_last_iter[loss_last_iter > best_loss ** conf_threshold],
               label='Rejected')
    ax.scatter(best_model, best_loss, label='Best')
    ax.plot([0, ensemble_size - 1], [best_loss ** conf_threshold, best_loss ** conf_threshold], linestyle='--')
    ax.set_xticks(np.arange(ensemble_size))
    ax.set_xlabel(r'Model number')
    ax.set_ylabel(r'Final loss')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(f"{output_dir}/{material}/models/loss.jpg")
    plt.close()


def loss_history_plot(ensemble_iter):
    data_train_history = pd.read_csv(f"{output_dir}/{material}/models/{ensemble_iter}/training_history.csv")
    loss = data_train_history['loss'].values / len(train_steps) / num_node
    eq_loss = data_train_history['eq_loss'].values / len(train_steps) / num_node
    bc_loss = data_train_history['bc_loss'].values / len(train_steps) / num_node
    stress_loss = data_train_history['stress_loss'].values / len(train_steps) / num_element
    learning_rate = data_train_history['lr'].values
    num_epoch = loss.shape[0]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(np.arange(1, num_epoch + 1), loss, label='Total loss', linestyle='-')
    ax1.plot(np.arange(1, num_epoch + 1), eq_loss, label='Internal loss', linestyle=':')
    ax1.plot(np.arange(1, num_epoch + 1), bc_loss, label='Boundary loss', linestyle=':')
    ax1.plot(np.arange(1, num_epoch + 1), stress_loss, label='PK stress loss', linestyle='-.')
    ax2.plot(np.arange(1, num_epoch + 1), learning_rate, label='Learning rate', linestyle='--')

    ax1.set_xlabel(r'Epoch number')
    ax1.set_ylabel(r'Loss value')
    ax2.set_ylabel(r'Learning rate')
    ax1.set_yscale('log')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax2.yaxis.set_major_formatter(formatter)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines],
               loc='center left', bbox_to_anchor=(1.0, 0.5), borderaxespad=4)
    fig.tight_layout()
    plt.savefig(f"{output_dir}/{material}/models/{ensemble_iter}/loss_history.jpg")
    plt.close()


def cauchy_comparison_subplot(ax, s_model, s_real, error, label):
    ax.scatter(s_real, s_model, label=fr'$\Delta={error * 100:.2f}\%$')
    ax.plot([torch.min(s_real), torch.max(s_real)], [torch.min(s_real), torch.max(s_real)],
            label='$y=x$', color='green')
    ax.legend()
    ax.set_xlabel(r'$\sigma_{' + label + r'}^\mathrm{FEM}\ (\mathrm{MPa})$')
    ax.set_ylabel(r'$\sigma_{' + label + r'}^\mathrm{ICNN}\ (\mathrm{MPa})$')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)


def nodal_force_surface_subplot(ax, coord, force, force_type=None, axis_label=None):
    coord = coord.cpu()
    force = force.cpu()

    # Interpolate data
    x_grid = torch.linspace(torch.min(coord[:, 0]), torch.max(coord[:, 0]), 100)
    y_grid = torch.linspace(torch.min(coord[:, 1]), torch.max(coord[:, 1]), 100)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    c_grid = griddata(points=(coord[:, 0], coord[:, 1]), values=force, xi=(x_grid, y_grid), method='cubic')
    contour = ax.pcolormesh(x_grid, y_grid, c_grid, shading='auto', cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax, shrink=0.7, orientation='horizontal')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.set_label(r'$' + force_type + r'\ (\mathrm{N})$')
    ax.set_xlabel(r'$' + axis_label[0] + r'\ (\mathrm{mm})$')
    ax.set_ylabel(r'$' + axis_label[1] + r'\ (\mathrm{mm})$')
    ax.set_aspect(aspect='equal', adjustable='box')


def nodal_force_3d_subplot(ax, coord, force, force_type=None):
    scatter = ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=force, cmap='viridis', s=40, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, orientation='horizontal')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.set_label(r'$' + force_type + r'\ (\mathrm{N})$')
    ax.set_xlabel(r'$X\ (\mathrm{mm})$')
    ax.set_ylabel(r'$Y\ (\mathrm{mm})$')
    ax.set_zlabel(r'$Z\ (\mathrm{mm})$')


def value_change_compare(ax, x_data, x_label, *y_info):
    color_list = ['blue', 'orange', 'green', 'red']
    linestyle_list = ['--', ':', '-.', '-']
    marker_list = ['o', 's', '^', 'D']
    if len(y_info) % 3:
        raise ValueError("The number of y information is not correct. Enter 'y_nn', 'y_real', 'y_label'.")
    y_number = len(y_info) // 3
    for i in range(y_number):
        ax.scatter(x_data, y_info[3 * i + 0], label='$' + y_info[3 * i + 2] + '^{NN}$',
                   color=color_list[i], marker=marker_list[i])
        ax.plot(x_data, y_info[3 * i + 1], label='$' + y_info[3 * i + 2] + '^{FE}$',
                color=color_list[i], linestyle=linestyle_list[i])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=y_number)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel(x_label)
    ax.set_ylabel('value')


def confidence_evaluation():
    # Confidence Evaluation
    loss_last_iter = np.zeros(ensemble_size)
    for ensemble_iter in range(ensemble_size):
        data_train_history = pd.read_csv(
            f"{output_dir}/{material}/models/{ensemble_iter}/training_history.csv")
        loss = data_train_history['loss'].values / len(train_steps) / num_node
        loss_last_iter[ensemble_iter] = loss[-1]
    best_model = np.argmin(loss_last_iter)
    best_loss = loss_last_iter[best_model]
    confidence = np.sum(loss_last_iter < best_loss ** conf_threshold) / ensemble_size
    ensemble_loss_plot(loss_last_iter, best_loss, best_model)
    print('-' * num_marker)
    print("Confidence evaluation:")
    print(f"Best model: {best_model}")
    print(f"Best loss: {best_loss: .6e}")
    print(f"Confidence: {confidence: .2%}")
    print('-' * num_marker)
    with open(f"{output_dir}/{material}/models/confidence.txt", 'w') as txt_file:
        txt_file.write(f"Best model: {best_model}\n")
        txt_file.write(f"Best loss: {best_loss: .6e}\n")
        txt_file.write(f"Confidence: {confidence: .2%}\n")


def evaluate_single_frame(ensemble_iter, model, data_type, load_step, mode=None):
    output_path = f"{output_dir}/{material}/{data_type}/{ensemble_iter}/{load_step}"
    os.makedirs(output_path, exist_ok=True)
    evaluate_data_path = get_data_path(data_dir, material)
    evaluate_data = load_data(evaluate_data_path, load_step, noise_type='none', noise_level=0.0)

    # Cauchy relative error
    pk_stress = compute_value(evaluate_data, model, type_name='pk_stress')
    sigma_model = pk_to_cauchy(evaluate_data.F, pk_stress)
    sigma_real = pd.read_csv(f"{evaluate_data_path}/{load_step}/stress.csv")
    sigma_real = torch.tensor(sigma_real[['sxx', 'syy', 'szz', 'sxy', 'sxz', 'syz']].values, dtype=torch.float)
    sxx_error, syy_error, szz_error, sxy_error, sxz_error, syz_error, s_mises_error = cauchy_error(sigma_model,
                                                                                                   sigma_real)
    print("Relative error for Cauchy stress:")
    print('-' * num_marker)
    print("|    sxx    |    syy    |    szz    |    sxy    |    sxz    |    syz    |  s_mises  |")
    print('-' * num_marker)
    print(f"|  {sxx_error:>7.2%}  |  {syy_error:>7.2%}  |  {szz_error:>7.2%}  "
          f"|  {sxy_error:>7.2%}  |  {sxz_error:>7.2%}  |  {syz_error:>7.2%}  "
          f"|  {s_mises_error:>7.2%}  |")
    print('-' * num_marker)
    with open(f"{output_path}/stress_relative_error", 'w') as txt_file:
        txt_file.write('-' * num_marker + '\n')
        txt_file.write("|    sxx    |    syy    |    szz    |    sxy    |    sxz    |    syz    |  s_mises  |\n")
        txt_file.write('-' * num_marker + '\n')
        txt_file.write(f"|  {sxx_error:>7.2%}  |  {syy_error:>7.2%}  |  {szz_error:>7.2%}  "
                       f"|  {sxy_error:>7.2%}  |  {sxz_error:>7.2%}  |  {syz_error:>7.2%}  "
                       f"|  {s_mises_error:>7.2%}  |\n")
        txt_file.write('-' * num_marker)

    # Cauchy stress comparison graph
    print("Drawing Cauchy stress comparison figures...", end='')
    sigma_model = sigma_model.detach()
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

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    cauchy_comparison_subplot(axs[0, 0], sxx_model, sxx_real, sxx_error, label='xx')  # sxx
    cauchy_comparison_subplot(axs[0, 1], syy_model, syy_real, syy_error, label='yy')  # syy
    cauchy_comparison_subplot(axs[0, 2], szz_model, szz_real, szz_error, label='zz')  # szz
    cauchy_comparison_subplot(axs[1, 0], sxy_model, sxy_real, sxy_error, label='xy')  # sxy
    cauchy_comparison_subplot(axs[1, 1], sxz_model, sxz_real, sxz_error, label='xz')  # sxz
    cauchy_comparison_subplot(axs[1, 2], syz_model, syz_real, syz_error, label='yz')  # syz
    cauchy_comparison_subplot(axs[2, 1], s_mises_model, s_mises_real, s_mises_error, label='mises')  # syz
    axs[2, 0].axis('off')
    axs[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_path}/stress_comparison.jpg")
    plt.close()
    print("Done.")

    # Nodal force graph for surfaces
    print("Drawing nodal force figures for surfaces...", end='')
    rf_node = compute_value(evaluate_data, model, type_name='nodal_force')
    rf_node = rf_node.detach().cpu()
    rf_real = pd.read_csv(f"{evaluate_data_path}/{load_step}/node_rf.csv")
    rf_real = torch.tensor(rf_real[['fx', 'fy', 'fz']].values, dtype=torch.float)
    for bc_rank in range(evaluate_data.num_bc):
        fig = plt.figure(figsize=(15, 20))
        node_rank = (evaluate_data.bc_node[:, bc_rank] == 1).cpu()
        x_node = evaluate_data.x_node[node_rank].cpu()
        if torch.all(x_node[:, 0] == x_node[0, 0]):
            draw_coord = x_node[:, [1, 2]]
            draw_label = ['Y', 'Z']
        elif torch.all(x_node[:, 1] == x_node[0, 1]):
            draw_coord = x_node[:, [0, 2]]
            draw_label = ['X', 'Z']
        elif torch.all(x_node[:, 2] == x_node[0, 2]):
            draw_coord = x_node[:, [0, 1]]
            draw_label = ['X', 'Y']
        else:
            warnings.warn("Not a flat surface.", UserWarning)
            continue
        ax = fig.add_subplot(4, 3, 1)
        nodal_force_surface_subplot(ax, draw_coord, rf_node[node_rank, 0], force_type=r'f_x^\mathrm{ICNN}',
                                    axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 2)
        nodal_force_surface_subplot(ax, draw_coord, rf_real[node_rank, 0], force_type=r'f_x^\mathrm{FEM}',
                                    axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 3)
        nodal_force_surface_subplot(ax, draw_coord, rf_node[node_rank, 0] - rf_real[node_rank, 0],
                                    force_type=r'f_x^\mathrm{ICNN}-f_x^\mathrm{FEM}', axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 4)
        nodal_force_surface_subplot(ax, draw_coord, rf_node[node_rank, 1], force_type=r'f_y^\mathrm{ICNN}',
                                    axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 5)
        nodal_force_surface_subplot(ax, draw_coord, rf_real[node_rank, 1], force_type=r'f_y^\mathrm{FEM}',
                                    axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 6)
        nodal_force_surface_subplot(ax, draw_coord, rf_node[node_rank, 1] - rf_real[node_rank, 1],
                                    force_type=r'f_y^\mathrm{ICNN}-f_y^\mathrm{FEM}', axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 7)
        nodal_force_surface_subplot(ax, draw_coord, rf_node[node_rank, 2], force_type=r'f_z^\mathrm{ICNN}',
                                    axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 8)
        nodal_force_surface_subplot(ax, draw_coord, rf_real[node_rank, 2], force_type=r'f_z^\mathrm{FEM}',
                                    axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 9)
        nodal_force_surface_subplot(ax, draw_coord, rf_node[node_rank, 2] - rf_real[node_rank, 2],
                                    force_type=r'f_z^\mathrm{ICNN}-f_z^\mathrm{FEM}', axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 10)
        nodal_force_surface_subplot(ax, draw_coord, torch.norm(rf_node[node_rank], dim=1),
                                    force_type=r'\left\Vert\boldsymbol{f}^\mathrm{ICNN}\right\Vert',
                                    axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 11)
        nodal_force_surface_subplot(ax, draw_coord, torch.norm(rf_real[node_rank], dim=1),
                                    force_type=r'\left\Vert\boldsymbol{f}^\mathrm{FEM}\right\Vert',
                                    axis_label=draw_label)
        ax = fig.add_subplot(4, 3, 12)
        nodal_force_surface_subplot(ax, draw_coord, torch.norm(rf_node[node_rank] - rf_real[node_rank], dim=1),
                                    force_type=r'\left\Vert\boldsymbol{f}^\mathrm{ICNN}-\boldsymbol{f}^\mathrm{FEM}\right\Vert',
                                    axis_label=draw_label)
        plt.tight_layout()
        plt.savefig(f"{output_path}/surface_{bc_rank}_force.jpg")
        plt.close()
    print("Done.")

    # Nodal force graph for test boundaries
    print("Drawing nodal force figures for test boundaries...", end='')
    for bc_rank in range(evaluate_data.num_test_bc):
        fig = plt.figure(figsize=(12, 16))
        node_rank = (evaluate_data.test_bc_node[:, bc_rank] == 1).cpu()
        x_node = evaluate_data.x_node[node_rank].cpu()
        ax = fig.add_subplot(4, 3, 1, projection='3d')
        nodal_force_3d_subplot(ax, x_node, rf_node[node_rank, 0],
                               force_type=r'f_x^\mathrm{ICNN}')
        ax = fig.add_subplot(4, 3, 2, projection='3d')
        nodal_force_3d_subplot(ax, x_node, rf_real[node_rank, 0],
                               force_type=r'f_x^\mathrm{FEM}')
        ax = fig.add_subplot(4, 3, 3, projection='3d')
        nodal_force_3d_subplot(ax, x_node, rf_node[node_rank, 0] - rf_real[node_rank, 0],
                               force_type=r'f_x^\mathrm{ICNN}-f_x^\mathrm{FEM}')
        ax = fig.add_subplot(4, 3, 4, projection='3d')
        nodal_force_3d_subplot(ax, x_node, rf_node[node_rank, 1],
                               force_type=r'f_y^\mathrm{ICNN}')
        ax = fig.add_subplot(4, 3, 5, projection='3d')
        nodal_force_3d_subplot(ax, x_node, rf_real[node_rank, 1],
                               force_type=r'f_y^\mathrm{FEM}')
        ax = fig.add_subplot(4, 3, 6, projection='3d')
        nodal_force_3d_subplot(ax, x_node, rf_node[node_rank, 1] - rf_real[node_rank, 1],
                               force_type=r'f_y^\mathrm{ICNN}-f_y^\mathrm{FEM}')
        ax = fig.add_subplot(4, 3, 7, projection='3d')
        nodal_force_3d_subplot(ax, x_node, rf_node[node_rank, 2],
                               force_type=r'f_z^\mathrm{ICNN}')
        ax = fig.add_subplot(4, 3, 8, projection='3d')
        nodal_force_3d_subplot(ax, x_node, rf_real[node_rank, 2],
                               force_type=r'f_z^\mathrm{FEM}')
        ax = fig.add_subplot(4, 3, 9, projection='3d')
        nodal_force_3d_subplot(ax, x_node, rf_node[node_rank, 2] - rf_real[node_rank, 2],
                               force_type=r'f_z^\mathrm{ICNN}-f_z^\mathrm{FEM}')
        ax = fig.add_subplot(4, 3, 10, projection='3d')
        nodal_force_3d_subplot(ax, x_node, torch.norm(rf_node[node_rank], dim=1),
                               force_type=r'\left\Vert\boldsymbol{f}^\mathrm{ICNN}\right\Vert')
        ax = fig.add_subplot(4, 3, 11, projection='3d')
        nodal_force_3d_subplot(ax, x_node, torch.norm(rf_real[node_rank], dim=1),
                               force_type=r'\left\Vert\boldsymbol{f}^\mathrm{FEM}\right\Vert')
        ax = fig.add_subplot(4, 3, 12, projection='3d')
        nodal_force_3d_subplot(ax, x_node,
                               torch.norm(rf_node[node_rank] - rf_real[node_rank], dim=1),
                               force_type=r'\left\Vert\boldsymbol{f}^\mathrm{ICNN}-\boldsymbol{f}^\mathrm{FEM}\right\Vert')
        plt.tight_layout()
        plt.savefig(f"{output_path}/test_boundary_{bc_rank}_force.jpg")
        plt.close()
    print("Done.")
    print('-' * num_marker)


def evaluate_single_element(ensemble_iter, model, bypass_model, single_element_data):
    output_path = f"{output_dir}/{material}/single_element/{ensemble_iter}"
    os.makedirs(output_path, exist_ok=True)
    evaluate_data_path = get_data_path(data_dir, material, single_element_data)
    sigma_model_list = []
    sigma_real_list = []
    energy_model_list = []
    energy_real_list = []
    num_load_step = 0

    for load_step in os.listdir(evaluate_data_path):
        if not os.path.isdir(os.path.join(evaluate_data_path, load_step)):
            continue
        evaluate_data = load_data(evaluate_data_path, load_step, noise_level=0.0, noise_type='displacement')
        num_load_step += 1

        pk_stress, strain_energy_model = compute_value(evaluate_data, model, type_name='pk_stress & energy')
        strain_energy_real = compute_value(evaluate_data, bypass_model, type_name='energy')
        sigma_model = pk_to_cauchy(evaluate_data.F, pk_stress)
        sigma_real = pd.read_csv(f"{evaluate_data_path}/{load_step}/stress.csv")
        sigma_real = torch.tensor(sigma_real[['sxx', 'syy', 'szz', 'sxy', 'sxz', 'syz']].values, dtype=torch.float)
        sigma_model_list.append(convert_tensor_to_numpy(sigma_model))
        sigma_real_list.append(convert_tensor_to_numpy(sigma_real))
        energy_model_list.append(strain_energy_model.item())
        energy_real_list.append(strain_energy_real.item())

    sigma_model_list = np.concatenate(sigma_model_list, axis=0)
    sigma_real_list = np.concatenate(sigma_real_list, axis=0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    x_label = r'Load step'
    value_change_compare(axs[0], np.arange(num_load_step), x_label,
                         sigma_model_list[:, 0], sigma_real_list[:, 0], r'\sigma_{xx}',
                         sigma_model_list[:, 1], sigma_real_list[:, 1], r'\sigma_{yy}',
                         sigma_model_list[:, 2], sigma_real_list[:, 2], r'\sigma_{zz}')
    value_change_compare(axs[1], np.arange(num_load_step), x_label,
                         sigma_model_list[:, 3], sigma_real_list[:, 3], r'\sigma_{xy}',
                         sigma_model_list[:, 4], sigma_real_list[:, 4], r'\sigma_{xz}',
                         sigma_model_list[:, 5], sigma_real_list[:, 5], r'\sigma_{yz}')
    value_change_compare(axs[2], np.arange(num_load_step), x_label,
                         energy_model_list, energy_real_list, 'W')
    plt.tight_layout()
    plt.savefig(f"{output_path}/{single_element_data}.jpg")
    plt.close()
