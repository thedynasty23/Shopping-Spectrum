# Shopping-Spectrum

A comprehensive customer segmentation and product recommendation system using machine learning techniques to analyze retail transaction data and provide intelligent shopping insights.

## 🚀 Project Overview

Shopping-Spectrum is an end-to-end data science project that combines customer analytics, machine learning clustering, and product recommendation systems. The project analyzes retail transaction data to segment customers into meaningful groups and provides personalized product recommendations based on purchasing patterns.

## ✨ Key Features

### 🔍 Customer Segmentation
- **RFM Analysis**: Recency, Frequency, and Monetary value analysis
- **K-Means Clustering**: Advanced clustering algorithm to identify customer segments
- **Customer Profiling**: Detailed insights into each customer segment
- **4 Customer Segments**:
  - **High-Value/Loyal Customers**: Frequent, high-spending customers
  - **Diverse/Recent Lapsed Browsers**: Exploratory customers needing reactivation
  - **Diverse, Steady, Mid-Value Buyers**: Occasional high-value purchasers
  - **Low Engagement/Reactivated Segment**: Infrequent customers requiring nurturing

### 🛍️ Product Recommendation System
- **Cluster-based Recommendations**: Recommendations based on customer segments
- **Collaborative Filtering**: Product suggestions based on similar customer behavior
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Real-time Recommendations**: Dynamic product suggestions

### 📊 Data Analysis & Visualization
- **Comprehensive EDA**: Transaction patterns, seasonal trends, customer behavior
- **Advanced Visualizations**: 3D cluster plots, radar charts, correlation heatmaps
- **Statistical Analysis**: Outlier detection, feature engineering, dimensionality reduction
- **Performance Metrics**: Silhouette score, clustering quality evaluation

## 🛠️ Technology Stack

- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework
- **Jupyter Notebook**: Development and analysis environment

## 📁 Project Structure

```
Shopping-Spectrum/
├── README.md                                    # Project documentation
├── Shopper_Spectrum.ipynb                      # Main analysis notebook
├── app.py                                       # Streamlit web application
├── requirements.txt                             # Python dependencies
├── kmeans_rfm_model.pkl                        # Trained clustering model
└── customer_data_with_recommendations.csv      # Customer data with recommendations
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/thedynasty23/Shopping-Spectrum.git
cd Shopping-Spectrum
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit application**
```bash
streamlit run app.py
```

4. **Access the application**
Open your browser and navigate to `http://localhost:8501`

## 💻 Usage

### Web Application Features

#### 🛍️ Product Recommendation
1. Select a product from the dropdown menu
2. Choose the number of recommendations (1-10)
3. Click "Recommend" to get similar products
4. View personalized product suggestions

#### 👥 Customer Segmentation
1. Input customer metrics:
   - **Recency**: Days since last purchase
   - **Frequency**: Number of orders
   - **Monetary**: Total spending amount
   - **Average Basket Value**: Average order value
   - **Tenure**: Customer relationship duration
   - **Returns**: Number of returned items
2. Click "Predict Segment" to classify the customer
3. View the predicted customer segment and characteristics

### Jupyter Notebook Analysis
- Open `Shopper_Spectrum.ipynb` for detailed analysis
- Complete data preprocessing and exploratory data analysis
- Customer segmentation methodology and evaluation
- Recommendation system development and testing

## 📈 Model Architecture

### Customer Segmentation Pipeline
1. **Data Preprocessing**: Cleaning, outlier detection, feature engineering
2. **Feature Engineering**: RFM metrics, behavioral insights, seasonality analysis
3. **Dimensionality Reduction**: PCA for optimal feature selection
4. **Clustering**: K-Means algorithm with optimal cluster determination
5. **Evaluation**: Silhouette analysis and cluster quality metrics

### Recommendation System
1. **Collaborative Filtering**: Customer-item interaction matrix
2. **Cosine Similarity**: Product similarity calculation
3. **Cluster-based Filtering**: Segment-specific recommendations
4. **Hybrid Approach**: Combining multiple recommendation strategies

## 📊 Key Results

- **4,078 customers** analyzed across 4 distinct segments
- **Silhouette Score**: 0.57 (indicating good cluster separation)
- **Calinski-Harabasz Score**: 20,115 (excellent cluster definition)
- **Davies-Bouldin Score**: 0.50 (moderate cluster separation)
- **10 personalized recommendations** per customer per segment

## 🎯 Business Impact

- **Customer Retention**: Identify at-risk customers for targeted campaigns
- **Revenue Optimization**: Focus on high-value customer segments
- **Personalization**: Deliver relevant product recommendations
- **Marketing Efficiency**: Segment-specific marketing strategies
- **Inventory Management**: Stock popular products by customer segment

## 🔧 Technical Details

### Data Processing
- **Missing Value Handling**: Strategic imputation and removal
- **Outlier Detection**: Isolation Forest algorithm
- **Feature Scaling**: StandardScaler for consistent data ranges
- **Feature Engineering**: 13 customer behavior features

### Machine Learning
- **Clustering Algorithm**: K-Means with optimal k=4
- **Model Validation**: Cross-validation and stability testing
- **Performance Optimization**: Efficient data structures and caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**thedynasty23**
- GitHub: [@thedynasty23](https://github.com/thedynasty23)
- Project Link: [https://github.com/thedynasty23/Shopping-Spectrum](https://github.com/thedynasty23/Shopping-Spectrum)

## 🙏 Acknowledgments

- Dataset inspiration from UCI Machine Learning Repository
- Streamlit community for excellent documentation
- Scikit-learn developers for robust ML algorithms
- Open source community for continuous inspiration

## 📱 Demo

Try the live application: [Shopping-Spectrum Demo](https://your-demo-link.streamlit.app)

---

**Built with ❤️ and data science**