import pandas as pd

# Step 1: Read the CSV file
csv_file = 'heat_map_source.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Step 2: Find the maximum x and y values
max_x = df['x'].max()
max_y = df['y'].max()

# Step 3: Divide the coordinate system into 9 squares (3x3 grid)
# Define the boundaries for each square
x_boundaries = [0, max_x / 3, 2 * max_x / 3, max_x]
y_boundaries = [0, max_y / 3, 2 * max_y / 3, max_y]

# Step 4: Calculate the frequency of points in each square
frequencies = [[0 for _ in range(3)] for _ in range(3)]

for index, row in df.iterrows():
    # print longitude and latitude as (long, lat) at precision 4
    # print(f"{row['longitude']:.4f},{row['latitude']:.4f}")
    x_index = sum(1 for boundary in x_boundaries if row['x'] > boundary) - 1
    y_index = sum(1 for boundary in y_boundaries if row['y'] > boundary) - 1
    frequencies[y_index][x_index] += 1

# Step 5: Display the frequencies
print("Frequencies of points in each square (3x3 grid):")
for i in range(3):
    for j in range(3):
        print(f"Square ({i + 1}, {j + 1}): {frequencies[i][j]} points")

# add new column 'aoi_requirement' to the dataframe
df['aoi_requirement'] = 0
# Step 6: Assign AOI requirements based on frequencies
for index, row in df.iterrows():
    x_index = sum(1 for boundary in x_boundaries if row['x'] > boundary) - 1
    y_index = sum(1 for boundary in y_boundaries if row['y'] > boundary) - 1
    frequency = frequencies[y_index][x_index]

    # Assign AOI requirement based on frequency
    if frequency > 150:
        df.at[index, 'aoi_requirement'] = 10
    elif 100 < frequency < 150:
        df.at[index, 'aoi_requirement'] = 15
    else:
        df.at[index, 'aoi_requirement'] = 20
# Step 7: Save the updated DataFrame to a new CSV file
output_file = 'real_world_trajectory.csv'  # Replace with the desired output file path
df.to_csv(output_file, index=False)
