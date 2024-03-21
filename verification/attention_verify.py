emergency_features = 3
embed_dim = 8
emergency_count = 3
batch_size = 32
total_samples = 3200
if __name__ == '__main__':
    import sys
    import numpy as np
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    # fix random seed (numpy and torch)
    np.random.seed(0)
    torch.manual_seed(0)
    num_heads = 1
    sys.path.append('..')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create embedding for mock_emergencies and agent_pos
    mock_emergencies = torch.rand((total_samples, emergency_count, 2)).to(device)
    agent_pos = torch.rand((total_samples, 1, 2)).to(device)
    query_network = nn.Linear(2, embed_dim).to(device)
    key_network = nn.Linear(emergency_features, embed_dim).to(device)
    value_network = nn.Linear(emergency_features, embed_dim).to(device)
    context_network = nn.Linear(embed_dim, emergency_count).to(device)

    predictor = nn.Sequential(
        nn.Linear(emergency_count * emergency_features, embed_dim),
        nn.Linear(embed_dim, embed_dim),
        nn.Linear(embed_dim, emergency_count)
    ).to(device)

    multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(device)
    # labels are one hot encoding that shows the closest emergency with agents
    distances = torch.norm(mock_emergencies - agent_pos, dim=-1)
    # Find the index of the smallest value
    min_index = torch.argmin(distances, axis=1)
    # Create a one-hot vector
    one_hot = torch.zeros_like(distances)
    one_hot[torch.arange(total_samples), min_index] = 1
    labels = torch.argmin(distances, dim=-1).to(torch.long).to(device)
    epochs = 80
    all_parameters = (list(predictor.parameters()))
    all_parameters = (list(query_network.parameters()) + list(key_network.parameters()) +
                      list(value_network.parameters()) + list(multihead_attn.parameters()))
    # + list(context_network.parameters()))
    optimizer = torch.optim.Adam(all_parameters, lr=1e-3)
    obs = torch.cat([mock_emergencies.reshape(-1, emergency_count, 2),
                     one_hot.unsqueeze(-1)], dim=-1).flatten(start_dim=1)
    inputs = torch.cat((obs, agent_pos.squeeze(1)), dim=1)
    dataset = torch.utils.data.TensorDataset(inputs.to(torch.float32), labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    progress = tqdm(range(epochs))
    criterion = nn.CrossEntropyLoss()
    predictor.train()
    query_network.train()
    key_network.train()
    value_network.train()
    multihead_attn.train()
    for _ in progress:
        mean_loss = 0
        for batch_observations, batch_distances in dataloader:
            # set mask to True for rows after i
            optimizer.zero_grad()
            emergency_pos, batch_agent_pos = torch.split(batch_observations, [emergency_count * emergency_features, 2],
                                                         dim=1)
            emergency_pos = emergency_pos.reshape(batch_size, emergency_count, emergency_features)
            batch_agent_pos = batch_agent_pos.unsqueeze(1)
            # emergency_pos, batch_agent_pos = torch.split(batch_observations.reshape(batch_size, -1, emergency_features),
            #                                              [emergency_count, 1], dim=1)
            query = query_network(batch_agent_pos)
            key = key_network(emergency_pos)
            value = value_network(emergency_pos)
            attn_output, attn_output_weights = multihead_attn(query, key, value)
            # calculate regress_loss of attn_output and labels
            # predicted_probs = predictor(batch_observations)
            loss = criterion(attn_output_weights.squeeze(1), batch_distances)
            # print example of predicted and actual labels
            print(attn_output_weights[:4])
            print(batch_distances[:4])
            loss.backward()
            mean_loss += loss.detach()
            optimizer.step()
        mean_loss /= len(dataloader)
        print("mean_loss: ", mean_loss.item())
        # progress.set_postfix({'mean_loss': mean_loss.item()})
        # increase the aoi of mock emergencies
    query_network.eval()
    key_network.eval()
    value_network.eval()
    multihead_attn.eval()
    # context_network.eval()
    # query = query_network(mock_emergencies)
    # key = key_network(agent_pos)
    # value = value_network(agent_pos)
    # attn_output, attn_output_weights = multihead_attn(query, key, value)
    # predicted_probs = context_network(attn_output)
    # print(attn_output_weights[:4])
    # predicted_probs = predictor(inputs)
    # selected_results = predicted_probs.argmax(dim=-1)
    # print accuracy
    # print((selected_results == labels).sum().item() / len(labels))
    # print(labels[:100])
