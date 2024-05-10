from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
import pickle

# Define the FastAPI app
app = FastAPI()

class InputData(BaseModel):
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: int
    Type: str
    
# Load the XGBoost model and scaler
with open('Final_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    

# Define the predict function
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

    # Return the prediction as a JSON response
    return {"prediction": prediction.item()}

# Run the FastAPI application using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
