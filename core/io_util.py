from .config import *


def get_data_path(data_dir, material, type):
    data_dir = f"{data_dir}/{material}/{type}"
    return data_dir


def export_tensor(folder, name, data, cols, header=True):
    os.makedirs(folder, exist_ok=True)

    # Export torch.tensor to csv
    data_frame = pd.DataFrame.from_records(data.detach().cpu().numpy())
    if header:
        data_frame.columns = cols
    data_frame.to_csv(f"{folder}/{name}.csv", header=header, index=False)


def export_array(folder, name, data, header=None):
    os.makedirs(folder, exist_ok=True)

    # Export torch.tensor to csv
    arr = np.array(data)
    np.savetxt(f"{folder}/{name}.csv", arr, delimiter=',', header=header, comments='')
