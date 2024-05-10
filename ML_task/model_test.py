import numpy as np
import pickle

# Example input features
sample_input = np.array([[297.0, 308.1, 1362,52.5,213,0,1,0]])  # Replace with your own sample input

with open("titanic\ML_task\Final_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("titanic\ML_task\scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Create a sample input data array
sample_input = np.array([[297.0, 308.1, 1362, 52.5, 213, 0, 1, 0]])

# Convert the sample input data array to the InputData format
input_data = InputData(
    Air_temperature=sample_input[0, 0],
    Process_temperature=sample_input[0, 1],
    Rotational_speed=sample_input[0, 2],
    Torque=sample_input[0, 3],
    Tool_wear=sample_input[0, 4],
    H=sample_input[0, 5],
    L=sample_input[0, 6],
    M=sample_input[0, 7],
)

# Call the predict function with the input data and print the output
prediction = predict(input_data)

# Print the prediction to verify the function works properly
print(f"Prediction: {prediction}")
