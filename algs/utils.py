import torch

_EPSILON = 1e-10  # small number to prevent indeterminate division


def normalize_batch(my_batch: torch.Tensor, normalize: bool = True):
    if normalize:
        normalized_batch = (my_batch - my_batch.mean(dim=(1, 2), keepdim=True)) / (
                my_batch.std(dim=(1, 2), keepdim=True) + torch.tensor(_EPSILON))
    else:
        normalized_batch = my_batch
    return normalized_batch


def shuffle_and_divide_data_dict(data: dict, n: int):
    """
    data: a dictionary containing the data for each policy
    n: number of parts to divide the data into
    """
    # Initialize empty dictionaries for the n parts
    divided_data = [{} for _ in range(n)]

    # Iterate over each key (e.g., 'car', 'drone') in the data
    for key in data.keys() - {'env_info'}:
        elements = data[key]

        # Ensure all tensors have the same first dimension size
        assert all(element.size(0) == elements[0].size(0) for element in elements)

        # Generate shuffled indices
        indices = torch.randperm(elements[0].size(0))

        # Calculate the size of each chunk
        chunk_size = elements[0].size(0) // n

        # Divide the data into n parts using the shuffled indices
        for i in range(n):
            start_idx = i * chunk_size
            end_idx = None if i == n - 1 else start_idx + chunk_size
            shuffled_indices = indices[start_idx:end_idx]

            divided_data[i][key] = tuple(element[shuffled_indices] for element in elements)

    return divided_data


import torch


def shuffle_and_divide_tuple(data_tuple: tuple, n: int):
    """
    data_tuple: a tuple containing different data types (e.g., actions, observations, rewards, dones)
    n: number of parts to divide the data into
    """
    # Ensure all tensors in the tuple have the same first dimension size
    assert all(tensor.size(0) == data_tuple[0].size(0) for tensor in data_tuple)

    # Initialize a list to hold the divided data
    divided_data = [[] for _ in range(n)]

    # Generate shuffled indices
    indices = torch.randperm(data_tuple[0].size(0))

    # Calculate the size of each chunk
    chunk_size = data_tuple[0].size(0) // n

    # Divide the data into n parts using the shuffled indices
    for i in range(n):
        start_idx = i * chunk_size
        end_idx = None if i == n - 1 else start_idx + chunk_size
        shuffled_indices = indices[start_idx:end_idx]

        # Append the shuffled and sliced data to the corresponding list
        for element in data_tuple:
            divided_data[i].append(element[shuffled_indices])

    # Convert each sublist in divided_data to a tuple
    divided_data = [tuple(sublist) for sublist in divided_data]

    return divided_data
