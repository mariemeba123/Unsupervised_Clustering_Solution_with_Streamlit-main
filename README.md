# Customer_Segmentation_application
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://unsupervisedclusteringsolutionwithapp-main-ebrawwgcgauuibap36s.streamlit.app/)

password - streamlit

This application allows businesses and marketers to segment their customers based on key attributes like age, annual income, and spending behavior. Using unsupervised machine learning (K-Means), it groups users into distinct clusters to assist in targeted marketing and strategy development.

## Features
- Interactive Streamlit interface with real-time cluster prediction.
- User-friendly form to input:
   - Age
   - Annual income
   - Spending score
- Visualization of clusters with user position overlaid.
- Silhouette plot to assess cluster quality.

## Dataset
The model is trained using the Mall Customers Dataset, containing:
- Age
- Annual Income (in $1000)
- Spending Score (1–100)

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model
We use the K-Means clustering algorithm to segment customers. It identifies groups of users with similar behaviors based on:
- Age
- Annual income
- Spending habits

## Future Enhancements
* Allow users to upload their own customer dataset.
* Enable filtering or selection of number of clusters (k).
* Integrate advanced clustering techniques like DBSCAN or Gaussian Mixture Models.
* Include customer profiles and actionable insights per cluster.

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/mariemeba123/Unsupervised_Clustering_Solution_with_Streamlit-main.git
   cd Unsupervised_Clustering_Solution_with_Streamlit-main

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run app.py

#### Thank you for using the  Customer Segmentation  Application! Feel free to share your feedback.
