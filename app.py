import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from PIL import Image
import io

with st.sidebar:
    st.header('Springboard - Capstone Project')
    st.write("Building a supervised learning model to predict the risk of breast cancer.")
    st.subheader("[Introduction](#predicting-the-risk-of-breast-cancer)")
    st.subheader("[Data Overview](#overview-of-the-data)")
    st.subheader("[Data Cleaning](#data-cleaning)")
    st.subheader("[Exploratory Data Analysis](#exploratory-data-analysis)")
    st.subheader("[Pre-Processing](#pre-processing)")
    st.subheader("[Modeling](#modeling)")
    st.subheader("[Next Steps](#next-steps)")
    st.write("Created and implemented by Joy Opsvig")

image = Image.open('src/headerimage2.jpg')
st.image(image)
st.caption('Photo from Unsplash')

st.markdown('''
# Predicting the Risk of Breast Cancer

Breast cancer is one of the most common types of cancer for women, 
    and it has the highest death rates among all cancers for women, second to lung cancer. 
    About 1 in 8 American women will develop invasive breast cancer over the course of her 
    lifetime (source: Breastcancer.org).

Early detection of malignant tumors is key to treating breast cancer patients. If breast cancer is found early on, 
    there are more treatment options available and higher chances of survival. 
    According to Carol Migard Breast Center,  “women whose breast cancer is detected 
    at an early stage have a 93 percent or higher survival rate in the first five years.”

I am going to build a classification model that can accurately identify the diagnosis 
    of breast cancer based on the measurements and attributes of a tumor.


### Source

The dataset was obtained from the University of Wisconsin Hospitals, Madison from Dr. William H. Wolberg: 
''')
st.write(" - [University of Wisconsin-Madison](https://pages.cs.wisc.edu/~olvi/uwmp/cancer.html)")

st.write(" - [Kaggle Source](https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset)")

st.markdown('''
### Data Science Method Approach

My approach to building a predictive algorithm is the following:
* Understand and review the attribute data of benign vs. malignant tumors.
* Identify any correlations between measurements and target variable (classification of tumor: benign vs. malignant).
* Build a model on cleaned data, separating training and test data.
* Perform model on test data and continue to improve model to achieve high accuracy rates.
* If my model performs with high accuracy, then I will be able to apply this model on measurements of new data and identify if tumors are at high risk of being classified as malignant.

''')

st.markdown('''
### Overview of the Data

Here is an overview of the first few lines of data. The data is comprised of measurements
    on tumors. We have five features, one response variable, and 569 unique entries. 
''')

with st.expander("Data Dictionary", expanded=False):
    st.write(
        """    
- **mean_radius**: mean of distances from center to points on the perimeter
- **mean_texture**: standard deviation of gray-scale values
- **mean_perimeter**: mean size of the core tumor
- **mean_area**: mean area size of the tumor
- **mean_smoothness**: mean of local variation in radius lengths
- **diagnosis**: dependent variable, the diagnosis of breast tissues (1 = malignant, 0 = benign)
   
        """
    )


df = pd.read_csv('src/breastcancer_data.csv')
st.dataframe(df.head())

st.markdown('''
And here are the summary statistics.
''')

st.dataframe(df.describe())

st.markdown('''
### Data Cleaning

''')

st.markdown('''

For cleaning, I want to check for correct data types, as well as for missing values, 
    potential outliers, any errors (typos, structural), potential duplicates, and the like. 
''')

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.markdown('''
It appears there are no missing values for any of the features and the data types are correct. 
    Let’s check on the distribution of each of the features.
''')

image2 = Image.open('src/featuredistribution.png')
st.image(image2)

st.markdown('''
The features 'mean_area' and 'mean_perimeter' slightly skew left, and the rest of features nearly 
    follow a gaussian distribution. We should be okay to use these features in model training.

The data is cleaned and ready for exploratory data analysis.
''')

st.markdown('''
### Exploratory Data Analysis

For my exploratory data analysis, I will aim to understand the relationship between variables
    and identify any potential correlations or insightful observations. 

Let's take a look at the boxplots comparing each of the features for a benign diagnosis (0) vs a malignant diagnosis (1).
''')

image3 = Image.open('src/boxplotsdiagnosis.png')
st.image(image3)

st.markdown('''
Reviewing the above charts, it appears benign tumors tend to have higher measurements 
    on average across all of the features.

It also appears there may be a potential outlier as one tumor 
    has a high 'mean_smoothness' measurement compared to the other entries. 
    After further investigation into this row of data, I made the decision to 
    remove the entry. The other measurements for the row were similar to other benign 
    measurements, but the smoothness was uncharacteristically high.

I would also like to understand how the features relate to each other, 
    comparing on a granular level how benign and malignant tumors also compare. 
    For this, I present the following pair plots.


''')

image4 = Image.open('src/pairplots.png')
st.image(image4)

st.markdown('''
The pair plots give a useful overview of how benign and malignant tumor measurements compare across all features.

I will now investigate any correlations between the features in the below heatmap.

''')

image5 = Image.open('src/heatmap.png')
st.image(image5)

st.markdown('''
As expected, 'mean_area', 'mean_radius', and 'mean_perimeter' are closely correlated with one another.

Based on the above exploratory data analysis, I predict that tumors with higher measurements compared 
    to the average (mean) measurements of the datasets are more likely to be benign than malignant.
''')

st.markdown('''
### Pre-Processing

Ahead of training my supervised learning model, I'll split my data into training and test sets, and then applying the
    same pre-processing techniques to each.

There are no categorical variables, so I did not have to one-hot encode my data. In order to normalize all the feature
    measurements on the same scale (from 0-1), I did apply normalization in the form of MinMaxScaler(). This is specifically
    important for distance-based modeling. 

Now that my data has been pre-processed, it is ready for model training.

''')

st.markdown('''
### Modeling

Knowing this is a binary classification supervised learning problem, 
    I am going to test a logistic regression model, random forest model, and K-nearest neighbors model.

For my modeling metrics, I will be looking at precision, recall, and the F1 score. Since my response variable is the diagnosis of
    a benign or malignant tumor, I want to avoid false-negatives at all costs -- meaning I want to avoid incorrectly
    identifying a malignant tumor as benign. For this, I will prioritize the recall score of my models.

''')

st.markdown('''
**Logistic Regression**

*Accuracy score: 0.89* 
''')
data = {'Precision': [0.97, 0.86], 
        'Recall': [0.70, 0.99],
        'F1-Score': [0.81, 0.92]}
logdata = pd.DataFrame(data)

st.dataframe(logdata)

st.markdown('''
**Random Forest**

*Accuracy score: 0.94* 
''')
fordata = {'Precision': [0.96, 0.88], 
        'Recall': [0.88, 0.98],
        'F1-Score': [0.92, 0.96]}
forestdata = pd.DataFrame(fordata)

st.dataframe(forestdata)

st.markdown('''
**K-Nearest Neighbors**

*Accuracy score: 0.91* 
''')
knndata = {'Precision': [0.93, 0.90], 
        'Recall': [0.80, 0.97],
        'F1-Score': [0.86, 0.93]}
neighbydata = pd.DataFrame(knndata)

st.dataframe(neighbydata)

st.markdown('''
In reviewing the three models above, my random forest model had the highest accuracy
    as well as a relatively high recall for malignant tumors (98%). Logistic regression
    also had a high recall measurement for malignant tumors (99%), however, my random forest
    model scored higher across the board of other metrics, so I would lean towards using this model.
''')

st.markdown('''
### Next Steps

As for next steps, I would consider conducting the following:
* Collecting more data entries, upwards of 1,000,000 rows or more, given my dataset is very small 
* I would work with the research team to better understand if there are additional features to measure and train the model on
* I would continue improving my model's accuracy by implementing hyperparameter tuning for my logisitic and random forest model

''')