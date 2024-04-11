import numpy as np
import pandas as pd
import streamlit as st 
import pickle
from sklearn import preprocessing
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train a KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Save the model
pickle.dump(knn, open('C:\\Users\\harsh\\OneDrive\\Desktop\\ML ASMT\\training_model.sav', 'wb'))

# Save the encoder separately
encoder_dict = {}
for i, name in enumerate(iris.feature_names):
    encoder_dict[name] = list(iris.target_names)
pickle.dump(encoder_dict, open('C:\\Users\\harsh\\OneDrive\\Desktop\\ML ASMT\\encoder_dict.pkl', 'wb'))

# Define the main function
def main(): 
    st.title("Iris Classifier")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Species Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    sepal_length = st.number_input("Sepal Length", 0.0)
    sepal_width = st.number_input("Sepal Width", 0.0) 
    petal_length = st.number_input("Petal Length", 0.0) 
    petal_width = st.number_input("Petal Width", 0.0) 

    if st.button("Predict"): 
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        knn_model = pickle.load(open('C:\\Users\\harsh\\OneDrive\\Desktop\\ML ASMT\\training_model.sav', 'rb'))
        prediction = knn_model.predict(features)
        st.success('Iris Species is {}'.format(iris.target_names[prediction[0]]))

if __name__=='__main__': 
    main()
