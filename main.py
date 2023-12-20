# Import libraries
import os
import pandas as pd
import re
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt
from fuzzywuzzy import process, fuzz
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Function to clean and format names
def clean_name(name):
    normalized_name = unicodedata.normalize('NFKC', name)
    name = normalized_name.replace('_', ' ')
    filtered_name = re.sub(r'[^a-zA-ZÀ-ÖØ-öø-ÿ \-]', '', name)
    joined_name = ' '.join(filtered_name.split())
    titled_name = joined_name.title()
    return titled_name

# Function to find similar names using fuzzy search
def find_similar_names(df, column_name='player_name', threshold=99):
    similar_names_dict = {}  
    names = df[column_name].tolist()  
    exclude_names = ['Kyle Walker', 'Kyle Walker-Peters'] 
    for name in names:
        if name in exclude_names:
            continue
        matches = process.extractBests(name, names, 
        scorer=fuzz.token_set_ratio, score_cutoff=threshold)
        filtered_matches = [match for match, score in matches 
        if match != name and match not in exclude_names and match 
        not in similar_names_dict]
        if filtered_matches:
            similar_names_dict[name] = filtered_matches
    return similar_names_dict


def tally_similar_names(similar_names_dict):
    return sum(len(matches) for matches in similar_names_dict.values())


# Load different csv files, combine them into one dataframe
dataset_dir = "/content/Dataset"
combined_df = pd.DataFrame()
for file_name in os.listdir(dataset_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(dataset_dir, file_name)
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        df['season'] = file_name.replace('.csv', '')
        combined_df = pd.concat([combined_df, df], ignore_index=True)



# Clean player names in the dataset
combined_df['clean_name'] = combined_df['name'].apply(clean_name)
player_names_df = pd.DataFrame(combined_df['clean_name'].unique(), 
columns=['player_name'])

# Apply fuzzy search to find and handle similar names
similar_names = find_similar_names(player_names_df)
similar_names = {key: value for key, value in similar_names.items() 
if 'Kyle Walker-Peters' not in value}

# Standardize names in the DataFrame
for original_name, similar_names_list in similar_names.items():
    for similar_name in similar_names_list:
        combined_df.loc[combined_df['clean_name'] == similar_name, 
        'clean_name'] = original_name


combined_df.loc[combined_df['clean_name'].isin(['Bobby De Cordova-Reid', 
'Bobby Reid']), 'clean_name'] = 'Bobby De Cordova-Reid'
player_names_df = pd.DataFrame(combined_df['clean_name'].unique(), 
columns=['player_name'])

total_similar_pairs = tally_similar_names(similar_names)
print(f"Total similar name pairs found: {total_similar_pairs}")

# Save the cleaned data to a CSV file
combined_df.to_csv('/content/updated_combined_data.csv', index=False)


# Aggregate different features 
aggregations = {
    'xP': 'mean',  
    'assists': 'sum',  
    'bonus': 'sum',  
    'bps': 'sum',  
    'clean_sheets': 'sum',  
    'creativity': 'mean',  
    'expected_assists': 'mean',  
    'expected_goal_involvements': 'mean',  
    'expected_goals': 'mean',  
    'expected_goals_conceded': lambda x: -x.mean(),  
    'goals_conceded': lambda x: -x.sum(), 
    'goals_scored': 'sum',  
    'ict_index': 'mean',  
    'influence': 'mean',  
    'minutes': 'sum',  
    'own_goals': 'sum',  
    'penalties_missed':lambda x: -x.sum(),  
    'penalties_saved': 'sum',  
    'red_cards': lambda x: -x.sum(),  
    'saves': 'sum',  
    'threat': 'mean',  
    'total_points': 'sum',  
    'value': 'mean',  
    'yellow_cards': lambda x: -x.sum(),  
    'attempted_passes': 'sum',  
    'big_chances_created': 'sum',  
    'big_chances_missed': lambda x: -x.sum(),  
    'clearances_blocks_interceptions': 'sum',  
    'completed_passes': 'sum',  
    'dribbles': 'sum',  
    'ea_index': 'first',  
    'errors_leading_to_goal':lambda x: -x.sum(),  
    'errors_leading_to_goal_attempt': lambda x: -x.sum(),  
    'fouls': lambda x: -x.sum(),  
    'key_passes': 'sum',  
    'offside': lambda x: -x.sum(),  
    'open_play_crosses': 'sum',  
    'penalties_conceded': lambda x: -x.sum(),  
    'recoveries': 'sum',  
    'tackled': 'sum',  
    'tackles': 'sum',  
    'target_missed': lambda x: -x.sum(),  
    'winning_goals': 'sum'  
}


# Aggregate player data by season and overall
df_grouped_season = combined_df.groupby(['clean_name', 'season']).
agg(aggregations).reset_index()
df_grouped_overall = combined_df.groupby('clean_name').agg(aggregations)
.reset_index()

df_descriptive = df_grouped_overall.copy()  

# Calculate and print descriptive statistics for 'total_points'
descriptive_stats = df_descriptive['total_points'].describe()
print(descriptive_stats)

# Print graphs
plt.figure(figsize=(10, 6))
sns.histplot(df_descriptive['total_points'], kde=False)
plt.title('Distribution of Total Points in Aggregated Data')
plt.xlabel('Total Points')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_descriptive['total_points'])
plt.title('Box Plot of Total Points in Aggregated Data')
plt.xlabel('Total Points')
plt.show()

# Calculate and print key statistics
descriptive_stats = df_grouped_overall['total_points'].describe()

median = descriptive_stats['50%']  
q1 = descriptive_stats['25%']  
q3 = descriptive_stats['75%']  
iqr = q3 - q1  
min_value = descriptive_stats['min']  
max_value = descriptive_stats['max']  

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"Median: {median}")
print(f"First Quartile (Q1): {q1}")
print(f"Third Quartile (Q3): {q3}")
print(f"Interquartile Range (IQR): {iqr}")
print(f"Minimum Value: {min_value}")
print(f"Maximum Value: {max_value}")
print(f"Lower Bound for Outliers: {lower_bound}")
print(f"Upper Bound for Outliers: {upper_bound}")


# Identify and analyze outliers in total points
q1 = df_descriptive['total_points'].quantile(0.25)  
q3 = df_descriptive['total_points'].quantile(0.75)  

iqr = q3 - q1  
outlier_condition = (df_descriptive['total_points'] < (q1 - 1.5 * iqr)) | 
(df_descriptive['total_points'] > (q3 + 1.5 * iqr))  
outliers = df_descriptive[outlier_condition]  
non_outliers = df_descriptive[~outlier_condition]  

total_points_sum = df_descriptive['total_points'].sum()  

non_outlier_total = non_outliers['total_points'].sum()  
outlier_total = outliers['total_points'].sum()  

outlier_q1 = outliers['total_points'].quantile(0.25)  
outlier_q3 = outliers['total_points'].quantile(0.75)  

outlier_iqr = outlier_q3 - outlier_q1  

outlier_outlier_condition = (outliers['total_points'] < (outlier_q1 - 
1.5 * outlier_iqr)) | (outliers['total_points'] > (outlier_q3 + 1.5 * outlier_iqr))  
outlier_outliers = outliers[outlier_outlier_condition]  


num_outliers = outliers.shape[0]  
num_non_outliers = non_outliers.shape[0]  
num_outlier_outliers = outlier_outliers.shape[0]  


print(f"Total outliers: {num_outliers}")
print(f"Outliers within outliers: {num_outlier_outliers}")
print(f"Total points of all players: {total_points_sum}")
print(f"Total points of non-outliers: {non_outlier_total}")
print(f"Total points of outliers: {outlier_total}")

# Print graphs
sns.boxplot(x=non_outliers['total_points'])
plt.title(f"Box Plot of Non-Outlier Total Points - Players: {num_non_outliers}")
plt.xlabel('Total Points')
plt.annotate(f"Players: {num_non_outliers}\nTotal Points: {non_outlier_total}"
, xy=(0.9, 0.9), xycoords='axes fraction', fontsize=10, ha='right', va='top')

plt.figure(figsize=(10, 6))
sns.boxplot(x=outliers['total_points'])
plt.title(f"Box Plot of Outlier Total Points - Players: {num_outliers}")
plt.xlabel('Total Points')
plt.annotate(f"Players: {num_outliers}\nTotal Points: {outlier_total}", 
xy=(0.9, 0.9), xycoords='axes fraction', fontsize=10, ha='right', va='top')

plt.show()


# Correlation analysis
df_eda = df_grouped_overall.copy() 
correlation_with_points = df_eda.corr()['total_points']
top_positive_features = correlation_with_points.sort_values(ascending=False)[1:6]
top_negative_features = correlation_with_points.sort_values(ascending=True)[:5]
combined_features = top_positive_features.index.union(top_negative_features.index)
selected_features = combined_features.insert(0, 'total_points')  
combined_correlation_matrix = df_eda[selected_features].corr()

# Print graphs
plt.figure(figsize=(12, 10))
sns.heatmap(combined_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Matrix with Top 5 Positive and 
Negative Correlated Features with Total Points')
plt.show()



# Analyze relationship between total points and other features
features = ['bps', 'goals_conceded']  
for feature in features:
    
    if feature in df_eda.columns:
        plt.figure(figsize=(10, 6))  
        sns.regplot(x=feature, y='total_points', data=df_eda, 
        line_kws={"color": "red"})  
        plt.title(f'Relationship between {feature} and Total Points')  
        plt.xlabel(feature)  
        plt.ylabel('Total Points')  
        x = df_eda[feature].dropna()  
        y = df_eda['total_points'][x.index]  
        r_squared = r2_score(y, np.poly1d(np.polyfit(x, y, 1))(x))  
        plt.text(0.05, 0.95, f'R² = {r_squared:.2f}', transform=plt.gca().transAxes, 
        fontsize=12, verticalalignment='top')  
        plt.show()  

# Train multivariable linear regression model
correlation_with_points = df_eda.corr(numeric_only=True)['total_points']
.sort_values(ascending=False)
top_features = correlation_with_points.index[1:6].tolist()
X = df_eda[top_features]
y = df_eda['total_points']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R-squared:", r2)


coefficients_table = pd.DataFrame({'Feature': top_features, 'Coefficient': model.coef_})
print(coefficients_table)

# Print graphs
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)


plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)


plt.xlabel('Actual Total Points')
plt.ylabel('Predicted Total Points')
plt.title('Actual vs Predicted Total Points')
plt.show()
