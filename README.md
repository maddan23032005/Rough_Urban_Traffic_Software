# Rough_Urban_Traffic_Software

This project is an AI-based traffic management system designed to reduce urban congestion using Multi-Agent Reinforcement Learning (MARL). The system is simulated using SUMO (Simulation of Urban MObility).

## Features
- Each traffic light is an independent AI agent.
- Agents learn to cooperate to optimize network-wide traffic flow.
- Built with Python, PyTorch, and SUMO.

## How to Run

1.  **Clone the repository:**
    `git clone https://github.com/maddan23032005/Rough_Urban_Traffic_Software.git` 
    `cd Rough_Urban_Traffic_Software`

2.  **Set up the environment:**
    `py -m venv venv`
    `.\venv\Scripts\Activate`
    `pip install -r requirements.txt`

3.  **Generate the SUMO network:**
    `cd simulation`
    `netconvert --node-files=map.nod.xml --edge-files=map.edg.xml --output-file=map.net.xml`
    `cd ..`

4.  **Train the AI Model:**
    `python main.py`

5.  **Evaluate the results:**
    `python run_evaluation.py`
    `python main_fixed.py`
    `python evaluate.py`