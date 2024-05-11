from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
import pickle
from pathlib import Path

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
class InputData(BaseModel):
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: int
    Type: str
    
with open(f'{BASE_DIR}/ML_task/Final_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open(f'{BASE_DIR}/ML_task/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    

@app.post('/predict')
async def predict(input_data: InputData):
    h=0;m=0;l=0
    if input_data.Type=='H':
        h=1
    elif input_data.Type=='M':
        m=1
    else:
        l=1
    input_array = [[
        input_data.Air_temperature,
        input_data.Process_temperature,
        input_data.Rotational_speed,
        input_data.Torque,
        input_data.Tool_wear,
        h,l,m
    ]]

    scaled_input = scaler.transform(input_array)
    print(scaled_input)
    prediction = model.predict(scaled_input)
    if prediction.item()==0:
        return {"prediction": "Heat Dissipation Failure"}
    elif prediction.item()==1:
        return {"prediction": "No Failure"}
    elif prediction.item()==2:
        return {"prediction": "Overstrain Failure"}
    elif prediction.item()==3:
        return {"prediction": "Power Failure"}
    elif prediction.item()==4:
        return {"prediction": "Random Failures"}
    else:
        return {"prediction": "Tool Wear Failure"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
