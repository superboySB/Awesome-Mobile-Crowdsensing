emergency_features = 3
embed_dim = 8
agent_count = 4
emergency_count = 128
if __name__ == '__main__':
    import sys
    import numpy as np
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    # fix random seed (numpy and torch)
    np.random.seed(0)
    torch.manual_seed(0)
    num_heads = 2
    sys.path.append('..')
    mock_emergencies = np.random.rand(emergency_count * emergency_features)
    aoi_indices = torch.arange(emergency_features - 1, emergency_count * emergency_features, emergency_features)
    mock_emergencies[aoi_indices] = 0
    agent_pos = np.random.rand(2 * agent_count)
    # create embedding for mock_emergencies and agent_pos
    mock_emergencies = torch.tensor(mock_emergencies, dtype=torch.float32).view(emergency_count, emergency_features)
    agent_pos = torch.tensor(agent_pos, dtype=torch.float32).view(agent_count, 2)
    query_network = nn.Linear(emergency_features, embed_dim)
    key_network = nn.Linear(2, embed_dim)
    value_network = nn.Linear(2, embed_dim)
    multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    emergency_matrix = mock_emergencies[..., :2].repeat(agent_count, 1)
    agent_matrix = agent_pos.repeat(1, emergency_count).view(-1, 2)
    labels = torch.norm(emergency_matrix - agent_matrix, dim=1).view(emergency_count, agent_count)
    # pass softmax to labels
    labels = nn.functional.softmax(labels, dim=1)
    epochs = 100
    all_parameters = (list(query_network.parameters()) + list(key_network.parameters()) +
                      list(value_network.parameters()) + list(multihead_attn.parameters()))
    optimizer = torch.optim.Adam(all_parameters, lr=1e-3)
    dataset = torch.utils.data.TensorDataset(mock_emergencies.to(torch.float32), labels.to(torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    progress = tqdm(range(epochs))
    criterion = nn.MSELoss()
    query_network.train()
    key_network.train()
    value_network.train()
    multihead_attn.train()
    for _ in progress:
        mean_loss = 0
        for batch_observations, batch_distances in dataloader:
            # set mask to True for rows after i
            optimizer.zero_grad()
            query = query_network(batch_observations)
            key = key_network(agent_pos)
            value = value_network(agent_pos)
            attn_output, attn_output_weights = multihead_attn(query, key, value)
            # calculate regress_loss of attn_output and labels
            regress_loss = criterion(attn_output_weights, batch_distances)
            regress_loss.backward()
            mean_loss += regress_loss.detach()
            optimizer.step()
        mean_loss /= len(dataloader)
        progress.set_postfix({'mean_loss': mean_loss.item()})
        # increase the aoi of mock emergencies
    query_network.eval()
    key_network.eval()
    value_network.eval()
    multihead_attn.eval()
    query = query_network(mock_emergencies)
    value = key = key_network(agent_pos)
    attn_output, attn_output_weights = multihead_attn(query, key, value)
    print(attn_output_weights[:4])
    print(labels[:4])
