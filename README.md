

For clone the project from the github use: 
open cmd then write: git clone https://github.com/shehab-bs1553/ML_Task_2.git

Main data is present in the archive folder.

Final processed data is present in the data folder named processed_data.csv

to run the API use : "uvicorn ML_task.Api:app --reload" in the terminal.

To run the docker container use: "uvicorn ML_task.Api:app --reload --port 8000 --host 0.0.0.0"

For building the docker image and run the container use : "docker compose up --build"