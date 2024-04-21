import pandas as pd

# Read the file
data = pd.read_csv('logs/bench-2024-04-21-02-32-20.txt', 
                   names=["test_case", "method", "success", "steps", "planning_time", "robot_time"],
                   header=None)

# Strip leading and trailing spaces
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert the success column to integer
data['success'] = data['success'].map({'True': 1, 'False': 0})

# Compute the counts for each group
counts = data.groupby(['test_case', 'method']).size().reset_index(name='counts')

# Compute the averages for each group
grouped = data.groupby(['test_case', 'method']).mean().reset_index()

# Merge the counts into the grouped DataFrame
grouped = pd.merge(grouped, counts, on=['test_case', 'method'])

# Pivot the table so each row corresponds to a single test case and each column corresponds to a particular statistic for a particular method
pivot = grouped.pivot(index='test_case', columns='method')

# Flatten the multi-level column index
pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]

# Reset the index to make 'test_case' a column again
pivot.reset_index(inplace=True)

# Save the DataFrame to a CSV file
pivot.to_csv('logs/bench-2024-04-21-02-32-20.csv', index=False)
