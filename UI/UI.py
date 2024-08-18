import streamlit as st
import pandas as pd
import numpy as np
import joblib as jl
import sklearn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from Ankun_Multicolumn import MultiColumnLabelEncoder, DataFrameSelector
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
import os

# Get the current file's directory
base_dir = os.path.dirname(__file__)

# Construct the full path to the CSV file
df_file_path = os.path.join(base_dir, 'Remove-null-car_name-and-fill-null.csv')
df3_file_path = os.path.join(base_dir, 'DataVersion3.csv')

bagging_model_file_path = os.path.join(base_dir, 'bagging_model.joblib')
ab_model_file_path = os.path.join(base_dir, 'AdaBoost_model.joblib')
svr_model_file_path = os.path.join(base_dir, 'best_model.joblib')
dt_model_file_path = os.path.join(base_dir, 'dt_model.joblib')
gb_model_file_path = os.path.join(base_dir, 'gb_model.joblib')
knn_model_file_path = os.path.join(base_dir, 'knn_model.joblib')
pr_model_file_path = os.path.join(base_dir, 'pr_model.joblib')
rf_model_file_path = os.path.join(base_dir, 'rf_model.joblib')
sc_model_file_path = os.path.join(base_dir, 'similar_car.joblib')
xgb_model_file_path = os.path.join(base_dir, 'XGBoost_model.joblib')
stack_model_file_path = os.path.join(base_dir, 'stack_model.joblib')

ankun_process_file_path = os.path.join(base_dir, 'Ankun_processing.pkl')
bagging_process_file_path = os.path.join(base_dir, 'bagging_processing.pkl')
knn_process_file_path = os.path.join(base_dir, 'knn_process_data.pkl')
pr_process_file_path = os.path.join(base_dir, 'poly_regress_processor.pkl')
svr_process_file_path = os.path.join(base_dir, 'process_data.pkl')
stack_process_file_path = os.path.join(base_dir, 'stack_processing.pkl')
cluster_process_file_path = os.path.join(base_dir, 'similar_process_data.pkl')



df = pd.read_csv(df_file_path)
df_ver3 = pd.read_csv(df3_file_path)

def main():    
    print("scikit-learn version:", sklearn.__version__)
    print("numpy version:", np.__version__)
    print("joblib version:", jl.__version__)
    html_temp="""
    <div style = "background-color: #a1d8ff; padding: 16px;">
    <h2 style="color: #4790f6; text-align:center;"> Simple Car Price Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write("")
    st.write("""            
    ##### Do you need a help with predicting your wanted car's price?
    ##### We will help you!
    ##### Answer our questions and the result will be showed to you!
            """)
    st.write("")
    
    input_feature = user_input_feature()
    bagging_processed_feature = processing(bagging_process_file_path,input_feature)
    Ankun_processed_feature = processing(ankun_process_file_path,input_feature)
    knn_processed_feature = processing(knn_process_file_path,input_feature)
    svr_processed_feature = processing(svr_process_file_path,input_feature)
    pr_processed_feature = processing(pr_process_file_path, input_feature)
    stack_processed_feature = processing(stack_process_file_path,input_feature)
    cluster_processed_feature = processing(cluster_process_file_path, input_feature)
    
    
    svr_model = jl.load(svr_model_file_path)
    rf_model = jl.load(rf_model_file_path)
    dt_model = jl.load(dt_model_file_path)
    knn_model = jl.load(knn_model_file_path)
    bagging_model = jl.load(bagging_model_file_path)
    ab_model = jl.load(ab_model_file_path)
    gb_model = jl.load(gb_model_file_path)
    xgb_model = jl.load(xgb_model_file_path)
    pr_model = jl.load(pr_model_file_path)
    stack_model = jl.load(stack_model_file_path)
    similar_car_model = jl.load(sc_model_file_path)

    try:
        if st.button("Seek for similar cars"):
            distances, indices = similar_car_model.kneighbors(cluster_processed_feature)
            print(indices[0])
            print(df.loc[indices[0],'car_name'])
        
            for i in indices[0]:    
                st.success(df.loc[i,'car_name'] +": "+df.loc[i,'url'] )
    except Exception as e:
        st.warning(e)

    try:
        if st.button("Predict"):
            rf_price_predict = float(rf_model.predict(Ankun_processed_feature))
            rf_price_predict_in_vnd = transform_to_vnd(rf_price_predict)
            dt_price_predict = float(dt_model.predict(Ankun_processed_feature))
            dt_price_predict_in_vnd = transform_to_vnd(dt_price_predict)
            knn_price_predict = float(knn_model.predict(knn_processed_feature))
            knn_price_predict_in_vnd = transform_to_vnd(knn_price_predict)
            svr_price_predict = float(svr_model.predict(svr_processed_feature))
            svr_price_predict_in_vnd = transform_to_vnd(svr_price_predict)
            bagging_price_predict = float(bagging_model.predict(bagging_processed_feature))
            bagging_price_predict_in_vnd = transform_to_vnd(bagging_price_predict)
            pr_price_predict = float(pr_model.predict(pr_processed_feature))*0.98
            pr_price_predict_in_vnd = transform_to_vnd(pr_price_predict)
            ab_price_predict = float(ab_model.predict(Ankun_processed_feature))
            ab_price_predict_in_vnd = transform_to_vnd(ab_price_predict)  
            gb_price_predict = float(gb_model.predict(Ankun_processed_feature))
            gb_price_predict_in_vnd = transform_to_vnd(gb_price_predict)       
            xgb_price_predict = float(xgb_model.predict(Ankun_processed_feature))
            xgb_price_predict_in_vnd = transform_to_vnd(xgb_price_predict)             
            stack_price_predict = float(stack_model.predict(stack_processed_feature))*0.95
            stack_price_predict_in_vnd = transform_to_vnd(stack_price_predict)           
                        
            st.balloons()
            
            st.success(f"Bagging model's prediction: {bagging_price_predict_in_vnd}")
            st.success(f"Polynomial Regression model's prediction: {pr_price_predict_in_vnd}")
            st.success(f"GradientBoost model's prediction: {gb_price_predict_in_vnd}")
            st.success(f"KNN's prediction: {knn_price_predict_in_vnd}")
            st.success(f"SVR model's prediction: {svr_price_predict_in_vnd}")
            st.success(f"AdaBoost model's prediction: {ab_price_predict_in_vnd}")
            st.success(f"Decision tree's prediction: {dt_price_predict_in_vnd}")
            st.success(f"Random forest's prediction: {rf_price_predict_in_vnd}")            
            st.success(f"Stacking models' prediction: {stack_price_predict_in_vnd}")
            st.success(f"XGBoost model's prediction: {xgb_price_predict_in_vnd}")

    except Exception as e:
        st.warning(e)
    

        
def user_input_feature():
    origin = st.selectbox("Which is your wanted car's origin?",(df['origin'].unique().tolist()))
    st.write("")

    car_model = st.selectbox("Which is your wanted car's model?",(df['car_model'].unique().tolist()))
    st.write("")
    car_name = st.selectbox("What is your wanted car's name?",(df['car_name'].unique().tolist()))
    st.write("")
    year_of_manufacture = int(st.number_input("Which year is that car produced?"))
    st.write("")
    mileage = st.number_input("How much mileage your wanted car had gone?")
    st.write("")
    exterior_color = st.selectbox("Which exterior color does that car have?", (df['exterior_color'].unique().tolist()))
    st.write("")
    interior_color = st.selectbox("Which interior color does that car have?",(df['interior_color'].unique().tolist()))
    st.write("")
    num_of_doors = st.number_input("How many doors does that car have?")
    st.write("")
    seating_capacity = st.number_input("How many seats does that car have?")
    st.write("")
    engine = st.selectbox("Which engine does that car use?", (df['engine'].unique().tolist()))
    st.write("")
    engine_capacity = st.number_input("How much capacity of that engine?")
    st.write("")
    fuel_consumption = st.number_input("How much fuel consumption does that car have?")
    st.write("")
    transmission = st.selectbox("Which type of transmission of that car?",(df['transmission'].unique().tolist()))
    st.write("")
    drive_type = st.selectbox("Which type of drive that car use?",(df["drive_type"].unique().tolist()))
    
    brand,grade = get_brand_grade(car_name)
    
    data = {
        'origin': origin,
        'car_model': car_model,
        'exterior_color': exterior_color,
        'interior_color': interior_color,
        'engine': engine,
        'transmission': transmission,
        'drive_type': drive_type,
        'brand': brand,
        'grade': grade,
        'car_name': car_name,
        'num_of_doors':num_of_doors,
        'seating_capacity': seating_capacity,
        'engine_capacity': engine_capacity,
        'fuel_consumption': fuel_consumption,
        'mileage': mileage,
        'year_of_manufacture': year_of_manufacture
    }
    features = pd.DataFrame(data, index=[0])
    return features    
    
def processing(trained_process,data):
    process_model = jl.load(trained_process)
    processed_data = process_model.transform(data)
    return processed_data

# Function to apply LabelEncoder to each categorical column
def label_encode_columns(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df

def transform_to_vnd(price_in_billion):
    price_in_vnd = price_in_billion * 10 ** 9
    return f"{price_in_vnd:,.0f} VND"

def get_brand_grade(car_name):
    record = df.loc[df['car_name'] == car_name, ['brand', 'grade']].iloc[0]
    return record['brand'],record['grade']


# Custom function to drop columns
def drop_columns(df, columns):
    return df.drop(columns, axis=1)

# Custom function to handle label encoding
def label_encode(df, columns):
    label_encoders = {}
    for column in columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

# Custom function for full preprocessing
def full_preprocessing(df):
    columns_for_encoding = ['origin', 'car_model', 'exterior_color', 'interior_color', 'engine', 'transmission', 'drive_type']
    df, label_encoders = label_encode(df, columns_for_encoding)
    df = df.dropna()
    return df

if __name__ == "__main__":
    main()
    