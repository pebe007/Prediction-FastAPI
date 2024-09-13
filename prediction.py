from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from starlette.exceptions import HTTPException
from fastapi.exceptions import RequestValidationError
from exception_handlers import request_validation_exception_handler, http_exception_handler, unhandled_exception_handler
from middleware import log_request_middleware
from sklearn.preprocessing import OneHotEncoder

# ngeload model serta one hot encodernya 
model = joblib.load('model.pkl')
encoder = joblib.load('label_encoders.pkl')

# Initialize FastAPI app
app = FastAPI()
app.middleware("http")(log_request_middleware)
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# inisialisasi column yang original 
original_columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

class ClientData(BaseModel): #constraint untuk fastapi
    age: int = Field(gt=0, le=100)
    job: str = Field(max_length=20, min_length=1, enum=['retired', 'services', 'admin.', 'management', 'technician', 'blue-collar', 'housemaid', 'self-employed', 'entrepreneur', 'unknown', 'student', 'unemployed', 'nan'])
    marital: str = Field(max_length=30, min_length=1, pattern=r"^[a-zA-Z]+$", enum=['divorced', 'married', 'single', 'unknown'])
    education: str = Field(max_length=30, min_length=1, enum=['basic.4y', 'high.school', 'university.degree', 'professional.course', 'basic.9y', 'unknown', 'basic.6y', 'illiterate'])
    default: str = Field(max_length=30, min_length=1, pattern=r"^[a-zA-Z]+$", enum=['yes','no'])
    housing: str = Field(max_length=30, min_length=1, pattern=r"^[a-zA-Z]+$", enum=['yes','no'])
    loan: str = Field(max_length=30, min_length=1, pattern=r"^[a-zA-Z]+$", enum=['yes','no'])
    contact: str = Field(max_length=30, min_length=1, pattern=r"^[a-zA-Z]+$", enum=['cellular', 'telephone'])
    month: str = Field(max_length=3, min_length=3, pattern=r"^[a-zA-Z]+$", enum=['nov', 'may', 'aug', 'jul', 'apr', 'jun', 'sep', 'oct', 'mar', 'dec'])
    day_of_week: str = Field(max_length=3, min_length=3, pattern=r"^[a-zA-Z]+$", enum=['tue', 'wed', 'thu', 'mon', 'fri'])
    duration: float = Field(gt=-0.1, description='Input 999 if client was not previously contacted')
    campaign: int = Field(gt=-1)
    pdays: int = Field(gt=-1)
    previous: int = Field(gt=-1)
    poutcome: str = Field(max_length=30, min_length=1, pattern=r"^[a-zA-Z]+$", enum=['success', 'nonexistent', 'failure'])

@app.get("/")
def read_root():
    return {"message": "2602080825_Phoebe Patricia Wibowo UAS Model Deployment"}

@app.post('/predict')
def predict(data: ClientData):
    try:
        #Mengubah ke dataframenya menjadi satu row
        df = pd.DataFrame([data.dict()])  
        print("Input DataFrame:\n", df)  # Debug statement
        
        #memastikan input dan training columns sama 
        categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
        numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous']

        # Mengubah categori menggunakan encoder yang dimasukan ke dalam pickle 
        transformed_col = encoder.transform(df[categorical_columns]).toarray()
        ohe_col_names = encoder.get_feature_names_out(categorical_columns)
        transformed_df = pd.DataFrame(transformed_col, columns=ohe_col_names)
        
        # Menggabung numeric dan yang di transformed
        numeric_features = df[numeric_columns].reset_index(drop=True)
        final_features = pd.concat([numeric_features, transformed_df], axis=1)
        
        # reindex sesuai column yang ada 
        model_features = final_features.reindex(columns=model.feature_names_in_, fill_value=0)
        print("Reindexed Model Features DataFrame:\n", model_features)  # Debug statement
        
        #  melakukan prediksi 
        prediction = model.predict(model_features)
        return {'prediction': prediction[0]}
    
    except Exception as e:
        print(f"Exception: {e}")  # Debug statement
        raise HTTPException(status_code=500, detail=str(e))

