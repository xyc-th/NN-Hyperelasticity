from core import *


def post_process():
    print('=' * num_marker)
    print("Evaluating ICNN ensemble...")
    os.makedirs(f"{output_dir}/{material}/models", exist_ok=True)
    confidence_evaluation()
    for ensemble_iter in range(ensemble_size):
        print('-' * num_marker)
        print(f"Evaluating model {ensemble_iter}.")

        # Loss history graph
        print("Drawing loss history figure...", end='')
        loss_history_plot(ensemble_iter)
        print("Done.")

        # Load model
        # Trained model
        model = ICNN(n_input=n_input, n_hidden=n_hidden, n_output=n_output,
                     use_dropout=use_dropout, dropout_rate=dropout_rate,
                     bypass=False).to(device)
        model.load_state_dict(
            torch.load(f"{output_dir}/{material}/models/{ensemble_iter}/params.pth",
                       weights_only=True, map_location=device))
        model.eval()
        # Bypass model
        model_ideal = ICNN(n_input=n_input, n_hidden=n_hidden, n_output=n_output,
                           use_dropout=use_dropout, dropout_rate=dropout_rate,
                           bypass=True).to(device)
        model_ideal.eval()

        # Validate in training set
        for load_step in train_steps:
            evaluate_single_frame(ensemble_iter, model, 'train', load_step, mode='val')

        # Test in testing set, change geometry
        for load_step in test1_steps:
            evaluate_single_frame(ensemble_iter, model, 'test1', load_step, mode='test')

        # Test in testing set, change geometry
        for load_step in test2_steps:
            evaluate_single_frame(ensemble_iter, model, 'test2', load_step, mode='test')

        # Single element test
        print("Drawing single element test figure for tensile load...")
        evaluate_single_element(ensemble_iter, model, model_ideal, 'single_shear')
        print("Done.")
        print("Drawing single element test figure for shear load...")
        evaluate_single_element(ensemble_iter, model, model_ideal, 'single_tensile')
        print("Done.")

    print("Evaluation Completed.")
    print('=' * num_marker)


if __name__ == '__main__':
    post_process()
