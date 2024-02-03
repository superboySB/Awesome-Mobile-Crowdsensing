import os
import pandas as pd


def get_allocation_by_agent(column_name: str, csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    agent_id = 0
    allocation_by_agent = []
    for i in range(4, 12, 2):
        first, second = str(i), str(i + 1)
        this_agent_allocation = df[[first, second]]
        this_agent_allocation = this_agent_allocation.rename(columns={first: 'x', second: 'y'})
        this_agent_allocation[column_name] = agent_id
        all_xy = this_agent_allocation.drop_duplicates()
        allocation_by_agent.append(all_xy)
        agent_id += 1
    return pd.concat(allocation_by_agent)


if __name__ == '__main__':
    csv_file = os.path.join("/workspace", "saved_data", 'trajectories', 'Chengdu_NN_easy_20240203-112923.csv')
    # Load the CSV file into a DataFrame
    allocation_by_agent_NN = get_allocation_by_agent('NN_agent_id', csv_file)
    csv_file = os.path.join("/workspace", "saved_data", 'trajectories', 'Chengdu_greedy_easy_20240203-112418.csv')
    allocation_by_agent_greedy = get_allocation_by_agent('greedy_agent_id', csv_file)
    merged_result = allocation_by_agent_NN.merge(allocation_by_agent_greedy, on=['x', 'y'], how='outer')
    print(merged_result)
