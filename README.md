# **SpotTrend: Popular Music Prediction Using Machine Learning**

## **Project Overview**
"SpotTrend" is a machine learning project that aims to predict the popularity of music tracks based on various audio features and metadata. The goal is to help music streaming platforms, artists, and producers make informed decisions by forecasting the potential success of songs. By analyzing patterns in musical attributes, this project seeks to determine which characteristics are likely to influence a track's popularity.

## **Features**
- **Exploratory Data Analysis (EDA)**: Understand the data distribution and key patterns.
- **Correlation Analysis**: Identify features that have a strong association with popularity.
- **Data Preprocessing**: Scaling and cleaning data for better model performance.
- **Machine Learning Model**: Use Random Forest Regression to predict popularity.
- **Hyperparameter Tuning**: Optimize the model using GridSearchCV.
- **Feature Importance**: Highlight which attributes contribute most to a track’s success.
- **Visualization**: Graphical insights into actual vs. predicted results, feature correlations, and data distributions.

## **Dataset**
The dataset consists of 227 music tracks, including:
- **Track Metadata**: Track name, artists, album name, release date, etc.
- **Audio Features**: Energy, danceability, loudness, acousticness, valence, tempo, and more.
- **Popularity Score**: Target variable indicating the track's popularity.

**Note:** Make sure the dataset file (`Spotify_data.csv`) is available in the same directory as the code.

## **Getting Started**

### **Prerequisites**
To run this project, you need to have the following Python libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **Project Structure**
- `SpotTrend.py`: Main code file containing the logic for EDA, data processing, model training, and evaluation.
- `Spotify_data.csv`: Dataset file (ensure this is present in the same directory).
- `README.md`: Project description and guide.

### **Running the Project**
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   ```
2. **Navigate to the project directory:**
   ```bash
   cd SpotTrend
   ```
3. **Run the script:**
   ```bash
   python SpotTrend.py
   ```

The script will load the dataset, preprocess it, train a Random Forest Regressor, and provide visualizations of the results.

## **Code Explanation**
### **1. Data Loading & Preprocessing**
- The dataset is loaded using Pandas, and unnecessary columns are dropped.
- The features are normalized to improve the performance of the machine learning models.

### **2. Exploratory Data Analysis (EDA)**
- Scatter plots show the relationship between various audio features and popularity.
- A correlation matrix visualizes the strength of the relationships between different attributes.

### **3. Feature Selection**
- The features showing a significant relationship with popularity (like energy, loudness, danceability) are selected for training the model.

### **4. Model Training & Hyperparameter Tuning**
- Random Forest Regression is used as the core algorithm for prediction.
- Hyperparameters are tuned using GridSearchCV for better accuracy.

### **5. Model Evaluation & Visualization**
- The model's performance is evaluated using Mean Squared Error (MSE) and R² score.
- Scatter plots visualize the actual vs. predicted popularity scores.
- Feature importance is plotted to provide insights into which attributes are most influential.

### **6. Insights**
- **Energy, loudness, and danceability** are positively correlated with popularity, suggesting that more energetic and lively tracks tend to be more popular.
- **Acousticness** has a negative correlation with popularity, indicating that tracks with higher acoustic content are generally less favored.

## **Sample Results**
![Actual vs. Predicted Popularity](path/to/actual_vs_predicted.png)
*The scatter plot indicates how closely the model's predictions align with the actual popularity values.*

## **Conclusion**
"SpotTrend" successfully demonstrates how machine learning can be used to analyze and predict the popularity of music tracks. The insights derived from this project can be valuable for music producers, streaming platforms, and marketers aiming to understand user preferences and trends.

## **Future Improvements**
- Experiment with different machine learning algorithms (e.g., XGBoost, Neural Networks).
- Include additional metadata like genre and artist popularity.
- Improve hyperparameter tuning with techniques like RandomizedSearchCV.

## **Author**
- **Aman Shrivastva** - ML Egnineer and Data Scientist


