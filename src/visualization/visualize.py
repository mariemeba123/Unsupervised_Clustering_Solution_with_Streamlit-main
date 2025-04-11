
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap', fontsize=16)
    plt.show()

def plot_silhouette(wss):
    # Now, plot the silhouette plot
    plt.figure(figsize=(10, 6))
    wss.plot(x='cluster', y='Silhouette_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Plot')
    
    plt.savefig("silhouette.png")

def cluster(df,income,spending):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="Annual_Income", y="Spending_Score", hue=df["Cluster_Label"], palette="viridis", s=100)
    plt.scatter(income, spending, color="red", s=200, edgecolors="black", label="You")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score")
    plt.title("Customer Segmentation")
    # Adjust the legend to make it smaller and place it outside
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderpad=1)
    plt.savefig("cluster.png", bbox_inches="tight")
   

# def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix'):
#     """
#     Plot the confusion matrix for the given true and predicted labels.
    
#     Args:
#         y_true (numpy.ndarray): Array of true labels.
#         y_pred (numpy.ndarray): Array of predicted labels.
#         classes (list): List of class labels.
#         normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.
#         title (str, optional): Title for the plot. Default is 'Confusion Matrix'.
#     """
#     cm = confusion_matrix(y_true, y_pred)
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.xlabel('Predicted', fontsize=12)
#     plt.ylabel('Actual', fontsize=12)
#     plt.title(title, fontsize=16)
#     plt.show()