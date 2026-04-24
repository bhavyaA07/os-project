# CPU Scheduling RL Dashboard

This project demonstrates a Reinforcement Learning approach to CPU Scheduling, compared against traditional algorithms (FCFS, SJF, SRTF, Round Robin).

## How to Run the Project

This repository includes automated setup scripts so that you can run the project on any system with Python installed, without manually configuring environments.

### Requirements
- Python 3.8 or higher installed on your system.

### On Windows
1. Double-click the **`run.bat`** file in this folder.
2. The script will automatically:
   - Create a localized Python virtual environment.
   - Install all required libraries (`flask`, `stable-baselines3`, `pandas`, etc.).
   - Start the local web server.
3. Open the URL provided in the terminal (usually `http://localhost:5000/`) in your web browser.

### On Mac/Linux
1. Open your terminal and navigate to the project directory.
2. Give the run script execution permissions:
   ```bash
   chmod +x run.sh
   ```
3. Execute the script:
   ```bash
   ./run.sh
   ```
4. Open the URL provided in the terminal (usually `http://localhost:5000/`) in your web browser.

## Using the Dashboard
Once the server is running and you have opened the webpage:
- If the simulation has not run yet, click the **"Run Simulation"** button. This will take a moment as it trains the RL model and evaluates traditional algorithms.
- After completion, you will see a full interactive dashboard with Gantt charts, metrics comparison, and training logs.
