#Libraries used: pandas, matplotlib, numpy, scikit-learn, hmmlearn (version 2.5), and dependencies

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn import hmm

pd.set_option('display.max_columns', None)
filepath = 'btc_4h_data_2018_to_2024-2024-10-10.csv'
df = pd.read_csv(filepath, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Num Trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
#Drop useless columns
df.drop(columns=['Close time', 'Quote asset volume', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], inplace=True)
#Drop useless row
df = df.drop(0)
#Set index to timestamp
df = df.set_index('Timestamp')
df.index = pd.to_datetime(df.index)
#Convert all values to floats
df = df.astype(float)
#Calculate returns and volatility
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(window=48).std()
#Drop all nan values
df.dropna(inplace=True)

#Context: Features are open price, high price, low price, close price, volume, number of trades, returns, and volatility during a specific timestamp.
#The dataset is btc data in 4 hour intervals so obviously were gonna be looking at a lot of bitcoin price data.
#summary statistics
print(df.describe())
#basic visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)

ax1.plot(df.index, df['Close'])
ax1.set_title('Bitcoin Price over Time')
ax1.set_ylabel('Price')
ax2.set_xlabel('Date')

ax2.plot(df.index, df['Returns'])
ax2.set_title('Bitcoin Returns over Time')
ax2.set_ylabel('Returns')
ax2.set_xlabel('Date')

plt.tight_layout()
plt.show()

#Setup the PCA
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])

#Calculate maxes and mins for each principal component
pca1_max_index = pca_df['PC1'].idxmax()
pca2_max_index = pca_df['PC2'].idxmax()
pca1_min_index = pca_df['PC1'].idxmin()
pca2_min_index = pca_df['PC2'].idxmin()

print("\nPC1 max:")
print(pca_df['PC1'].iloc[pca1_max_index])
print(df.iloc[pca1_max_index])

print("\nPC2 max:")
print(pca_df['PC2'].iloc[pca2_max_index])
print(df.iloc[pca2_max_index])

print("\nPC1 min:")
print(pca_df['PC1'].iloc[pca1_min_index])
print(df.iloc[pca1_min_index])

print("\nPC2 min:")
print(pca_df['PC2'].iloc[pca2_min_index])
print(df.iloc[pca2_min_index])

#Plot unclustered PCA graph
plt.figure(figsize=(10, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.4)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D PCA Visualization (Unclustered)')
plt.show()

#Find PCA loadings
loadings = pd.DataFrame(pca.components_, columns=df.columns, index=['PC1', 'PC2'])
print(f'PCA Loadings:\n {loadings}')

#Find the features with the largest impact on each PC
for pc in loadings.index:
    most_important_num = 0
    most_important_feature = ''
    for i in range(len(loadings.loc[pc])-1):
        if abs(loadings.loc[pc][i]) > most_important_num:
            most_important_num = abs(loadings.loc[pc][i])
            most_important_feature = loadings.loc[pc].index[i]
    print(f'The feature with the biggest impact on {pc} is: {most_important_feature}')

#Cluster dataset and plot the clustered PCA visualiaztion
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_df)
pca_df['Cluster'] = cluster_labels

plt.figure(figsize=(10, 6))
for cluster in np.unique(cluster_labels):
    cluster_data = pca_df[pca_df['Cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D PCA Visualization (Clustered)')
plt.legend()
plt.show()

#Print out a cluster summary to better understand what each cluster represents
df['Cluster'] = cluster_labels
cluster_summary = df.groupby('Cluster').mean()
print(f'Cluster Summary: \n{cluster_summary}')

#Bitcoin HMM analysis
#Get feature data and scale data
features = ["Close", "Returns", "Volatility"]
feature_data = df[features].values
feature_data_scaled = scaler.fit_transform(feature_data)

#Setup hmm model and predict states
hmm_model = hmm.GaussianHMM(n_components=4, covariance_type='full', n_iter=400, random_state=42)
hmm_model.fit(feature_data_scaled)
states = hmm_model.predict(feature_data_scaled)
df['State'] = states

#Analyze each state to better understand what each state represents
for state in range(4):
        print(f"Analyzing State {state}:")
        state_data = df[df['State'] == state]
        print(state_data[features].describe())
        print(f"Number of periods in State {state}: {len(state_data)}")

#Plot bitcoin price and associated states
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'])
plt.title('Bitcoin Price and HMM States')
plt.ylabel('Price')

for state in range(4):
        mask = (states == state)
        plt.fill_between(df.index, df['Close'].min(), df['Close'].max(), where=mask, alpha=0.3, label=f'State {state}')

plt.legend()
plt.show()