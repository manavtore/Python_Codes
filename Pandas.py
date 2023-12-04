import pandas as pd

# 1. Reading data from a CSV file into a DataFrame
df = pd.read_csv('data.csv')

# 2. Displaying the first few rows of a DataFrame
print(df.head())

# 3. Checking the basic information about the DataFrame
print(df.info())

# 4. Describing the statistical summary of numerical columns
print(df.describe())

# 5. Selecting a single column from the DataFrame
column_data = df['Column_Name']

# 6. Filtering rows based on a condition
filtered_data = df[df['Column_Name'] > 50]

# 7. Handling missing values by dropping or filling them
df_cleaned = df.dropna()  # or df.fillna(value)

# 8. Grouping data based on a column and applying an aggregate function
grouped_data = df.groupby('Column_Name').mean()

# 9. Sorting DataFrame by one or more columns
sorted_df = df.sort_values(by=['Column1', 'Column2'], ascending=[True, False])

# 10. Adding a new column based on existing columns
df['New_Column'] = df['Column1'] + df['Column2']

# 11. Merging two DataFrames based on a common column
merged_df = pd.merge(df1, df2, on='Common_Column')

# 12. Reshaping data using the pivot table
pivot_table = df.pivot_table(values='Value', index='Index_Column', columns='Column_Name', aggfunc='mean')

# 13. Applying a function element-wise to a column
df['New_Column'] = df['Column'].apply(lambda x: x * 2)

# 14. Removing duplicate rows based on specified columns
df_no_duplicates = df.drop_duplicates(subset=['Column1', 'Column2'])

# 15. Creating a cross-tabulation table
cross_tab = pd.crosstab(df['Column1'], df['Column2'])

# 16. Changing data types of columns
df['Column'] = df['Column'].astype('float')

# 17. Performing element-wise operations on columns
df['New_Column'] = df['Column1'] * df['Column2']

# 18. Handling date and time data
df['Date_Column'] = pd.to_datetime(df['Date_Column'])
df['Month'] = df['Date_Column'].dt.month

# 19. Applying a function to each element in a column
df['New_Column'] = df['Column'].apply(lambda x: custom_function(x))

# 20. Saving DataFrame to a CSV file
df.to_csv('output.csv', index=False)


# 20. Creating a new DataFrame by selecting specific columns
selected_columns_df = df[['Column1', 'Column2']]

# 21. Applying a function to each row in a DataFrame
df['New_Column'] = df.apply(lambda row: custom_function(row['Column1'], row['Column2']), axis=1)

# 22. Renaming columns
df.rename(columns={'Old_Name': 'New_Name'}, inplace=True)

# 23. Extracting unique values from a column
unique_values = df['Column'].unique()

# 24. Replacing values in a column
df['Column'] = df['Column'].replace({'old_value': 'new_value'})

# 25. Creating a new DataFrame by filtering based on multiple conditions
filtered_data_multiple_conditions = df[(df['Column1'] > 50) & (df['Column2'] < 100)]

# 26. Shuffling the rows of a DataFrame
shuffled_df = df.sample(frac=1)

# 27. Filling missing values using forward fill or backward fill
df_ffill = df.fillna(method='ffill')
df_bfill = df.fillna(method='bfill')

# 28. Calculating the cumulative sum of a column
df['Cumulative_Sum'] = df['Column'].cumsum()

# 29. Extracting a subset of rows and columns based on labels
subset = df.loc[1:5, ['Column1', 'Column2']]

# 30. Melting a DataFrame (converting wide format to long format)
melted_df = pd.melt(df, id_vars=['ID'], value_vars=['Month1', 'Month2'], var_name='Month', value_name='Value')

# 31. Applying a custom function to each element in a DataFrame
df.applymap(lambda x: custom_function(x) if pd.notnull(x) else x)

# 32. Checking for duplicate rows in the entire DataFrame
duplicate_rows = df.duplicated()

# 33. Resetting the index of a DataFrame
df_reset_index = df.reset_index()

# 34. Creating a new column based on a condition using np.select()
conditions = [df['Column1'] > 0, df['Column1'] <= 0]
choices = ['Positive', 'Non-Positive']
df['Category'] = np.select(conditions, choices)

# 35. Finding the index of the maximum value in a column
max_index = df['Column'].idxmax()

# 36. Applying a rolling window function to a column
df['Rolling_Avg'] = df['Column'].rolling(window=3).mean()

# 37. Converting a DataFrame to a NumPy array
numpy_array = df.to_numpy()

# 38. Combining two DataFrames vertically (concatenation)
concatenated_df = pd.concat([df1, df2], axis=0)

# 39. Applying a function to groups using groupby and transform
df['Group_Avg'] = df.groupby('Group')['Value'].transform('mean')

# 40. Creating a time series DataFrame with date range
date_range_df = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
time_series_df = pd.DataFrame({'Date': date_range_df, 'Value': np.random.randn(len(date_range_df))})