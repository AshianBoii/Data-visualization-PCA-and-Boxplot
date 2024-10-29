import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# load the MNIST data
mnist_data = pd.read_csv('MNIST_100.csv')

# Task 1: Visualize the MNIST Data Using PCA

# separates the labels and the pixel data
labels = mnist_data['label']
pixels = mnist_data.drop(columns=['label'])

# apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
mnist_pca = pca.fit_transform(pixels)

# creates a DataFrame with the reduced dimensions and the labels
mnist_pca_df = pd.DataFrame(data=mnist_pca, columns=['Principal Component 1', 'Principal Component 2'])
mnist_pca_df['Label'] = labels

# plotting the data
plt.figure(figsize=(10, 8))
sns.scatterplot(data=mnist_pca_df, x='Principal Component 1', y='Principal Component 2', hue='Label', palette='tab10')
plt.title('MNIST Data Visualization with PCA (2D)')
plt.show()

#Task 2: Visualize columns K, M, and N in the housing data using Boxplot

# load the housing data
housing_data = pd.read_csv('housing_training.csv')
# checks the column names in the housing dataset and prints them 
print(housing_data.columns) #prints the names of the columns that I then used to find out which columns were K,M,N
# column names for k,m,n the first value of the column
kcolumn = '15.3'
mcolumn = '4.98'
ncolumn = '24'

# creating a boxplot for the corrected columns
plt.figure(figsize=(12, 6))
sns.boxplot(data=housing_data[[kcolumn, mcolumn, ncolumn]])
plt.title('Boxplot of Columns K, M, and N')
plt.show()


# Task 3: Visualize the Column of A in the Housing Data Using Histogram


a_column = '0.00632'  # this is column a the first value

# creates a histogram for column A
plt.figure(figsize=(8, 6))
plt.hist(housing_data[a_column], bins=30, edgecolor='black')
plt.title('Histogram of Column A')
plt.xlabel(a_column)
plt.ylabel('Frequency')
plt.show()