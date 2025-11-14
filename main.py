#importing all classes
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
    num_pipeline= Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    cat_pipeline= Pipeline([
        ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline=ColumnTransformer([
        ("nums",num_pipeline,num_attribs),
        ("cats",cat_pipeline,cat_attribs)
    ])
    return full_pipeline


if not os.path.exists(MODEL_FILE):
    housing=pd.read_csv("housing.csv")
    housing["income_cat"]=pd.cut(housing["median_income"],
                                bins=[0.0,1.5,3.0,4.5,6.0,np.inf]
                                ,labels=[1,2,3,4,5])

    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(housing,housing['income_cat']):
        df=housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv")
        housing=housing.loc[train_index].drop("income_cat",axis=1)

    #working on training set
    housing_labels=housing['median_house_value'].copy()
    housing_features=housing.drop('median_house_value',axis=1)
    #print(housing,housing_labels)

    #separating numerical and categorical values
    num_attribs=housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attribs=["ocean_proximity"]

    pipeline=build_pipeline(num_attribs,cat_attribs)
    housing_prep=pipeline.fit_transform(housing_features)

    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prep,housing_labels)
    #saving model and dump
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("model trained succesfully")

else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)

    input_data=pd.read_csv("input.csv")
    actual = input_data["median_house_value"]
    trans_input=pipeline.transform(input_data)
    predictions=model.predict(trans_input)
    input_data["median_house_value"]=predictions
    input_data.to_csv("output.csv",index=False)
    print("Inference complete ,values predicted!!!")

    rmse = root_mean_squared_error(actual, predictions)
    print(f"Test set RMSE (accuracy): {rmse:.2f}")