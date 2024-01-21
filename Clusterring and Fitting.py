import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('CC GENERAL.csv')

data = data.drop(['CUST_ID'], axis=1)

data = data.fillna(data.mean())

data.head()

data.info()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

k_optimal = 3

kmeans = KMeans(n_clusters=k_optimal, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_data)
data['Cluster'] = clusters
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 6))
for cluster in range(k_optimal):
    plt.scatter(data[data['Cluster'] == cluster]['PCA1'],
                data[data['Cluster'] == cluster]['PCA2'],
                label=f'Cluster {cluster + 1}')
plt.title('Clusters Visualized in 2D')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=data.columns[:-3])

print("Cluster Centers:")
print(cluster_centers_df)

cluster_summary = data.groupby('Cluster').agg({
    'BALANCE': 'mean',
    'PURCHASES': 'mean',
    'CASH_ADVANCE': 'mean',
    'CREDIT_LIMIT': 'mean',
    'PAYMENTS': 'mean',
    'PRC_FULL_PAYMENT': 'mean',
    'MINIMUM_PAYMENTS': 'mean',
    'TENURE': 'mean'
}).rename(columns={'CUST_ID': 'Number of Customers'})

print("\nCluster Summary:")
print(cluster_summary)

metrics_to_visualize = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS','TENURE']
for metric in metrics_to_visualize:
    plt.figure(figsize=(10, 6))
    for cluster in range(k_optimal):
        plt.hist(data[data['Cluster'] == cluster][metric], bins=30, alpha=0.5, label=f'Cluster {cluster + 1}')

    plt.title(f'{metric} Distribution Across Clusters')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()