# ===================================
# Initialization
# ===================================
from core import *
from drivers.post import post_process

start_time = datetime.now()
print(f"Training starts at {start_time.strftime('%Y-%m-%d %H:%M:%S')}.")

# ===================================
# Data
# ===================================
# Initialize input data for training
train_datasets = []

# Load training data
train_data_path = get_data_path(data_dir, material, type='train')
for load_step in train_steps:
    train_data = load_data(train_data_path, load_step, noise_type=noise_type, noise_level=noise_level)
    train_datasets.append(train_data)
train_data_load_time = datetime.now()
print(f"Time to load data: {(train_data_load_time - start_time).total_seconds()} s.")

# ===================================
# Model
# ===================================
# Initialize model
model = ICNN(n_input=n_input, n_hidden=n_hidden, n_output=n_output,
             use_dropout=use_dropout, dropout_rate=dropout_rate).to(device)
print_nn_architecture(model)
model_init_time = datetime.now()
print(f"Time to initialize model: {(model_init_time - train_data_load_time).total_seconds()} s.")

# Train model
print('=' * num_marker)
print("Begin training.")
print(f"Training an ensemble of {ensemble_size} models...")
os.makedirs(f"{output_dir}/{material}/", exist_ok=True)
for ensemble_iter in range(ensemble_size):
    ensemble_train_start_time = datetime.now()
    if random_init_linear:
        model.apply(init_weights)
    print(f"Training model {ensemble_iter + 1} out of {ensemble_size}.")
    (model,
     loss_history, eq_loss_history, bc_loss_history, stress_loss_history,
     lr_history) = train_model(model, train_datasets)
    training_history = np.array([loss_history, eq_loss_history, bc_loss_history, stress_loss_history, lr_history])
    os.makedirs(f"{output_dir}/{material}/model/{ensemble_iter}", exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/{material}/model/{ensemble_iter}/params.pth")
    export_array(f"{output_dir}/{material}/model/{ensemble_iter}", f"training_history",
                 training_history.T, header='loss,eq_loss,bc_loss,stress_loss,lr')
    ensemble_train_end_time = datetime.now()
    print(f"Time usage: {(ensemble_train_end_time - ensemble_train_start_time).total_seconds()} s.")
print("End training.")
print('=' * num_marker)
model_train_time = datetime.now()
print(f"Time to train model: {(model_train_time - model_init_time).total_seconds()} s.")

# Validate and test model
post_process()

end_time = datetime.now()
print(f"Time to evaluate model: {(end_time - model_train_time).total_seconds()} s.")
print(f"Training ends at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
      f"total time: {(end_time - start_time).total_seconds()} s.")
