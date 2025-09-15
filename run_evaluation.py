import os
import sys
import traci
from sumolib import checkBinary
from agent.dqn_agent import DQNAgent # Import your agent
import numpy as np

# --- Helper function 'get_state' must be defined before it's used ---
def get_state(traffic_light_id, state_size):
    lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
    # This logic must match the state representation your model was trained on
    state = [
        max(queue_lengths[0], queue_lengths[2]),
        max(queue_lengths[1], queue_lengths[3])
    ]
    while len(state) < state_size:
        state.append(0)
    return np.array(state[:state_size])

# --- Main evaluation logic ---
def evaluate():
    # --- Load the saved models ---
    STATE_SIZE = 4
    ACTION_SIZE = 2
    TRAFFIC_LIGHT_IDS = ["J1", "J2"]

    agents = {tl_id: DQNAgent(STATE_SIZE, ACTION_SIZE, tl_id) for tl_id in TRAFFIC_LIGHT_IDS}
    for tl_id in TRAFFIC_LIGHT_IDS:
        try:
            agents[tl_id].load(f"{tl_id}_model.pth")
            agents[tl_id].epsilon = 0.0 # Turn off random actions for evaluation
            print(f"Successfully loaded model for {tl_id}")
        except FileNotFoundError:
            print(f"Error: Model file for {tl_id} not found. Make sure training is complete.")
            sys.exit(1)


    # --- SUMO Configuration ---
    SUMO_BINARY = checkBinary('sumo')
    CONFIG_FILE = "simulation/config.sumocfg"
    
    # --- Run a single evaluation episode ---
    print("\nStarting evaluation run with trained models...")
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--tripinfo-output", "tripinfo_trained_ai.xml"])

    step = 0
    while step < 3600:
        current_states = {tl_id: get_state(tl_id, STATE_SIZE) for tl_id in TRAFFIC_LIGHT_IDS}
        actions = {tl_id: agents[tl_id].act(current_states[tl_id]) for tl_id in TRAFFIC_LIGHT_IDS}

        for tl_id, action in actions.items():
            if action == 1:
                current_phase = traci.trafficlight.getPhase(tl_id)
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
                num_phases = len(logic[0].phases)
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(tl_id, next_phase)
        
        # We step 10 times to match the training environment's action frequency
        for _ in range(10):
            traci.simulationStep()
            step += 1
            
    traci.close()
    print("Evaluation finished. Results saved to 'tripinfo_trained_ai.xml'")


if __name__ == "__main__":
    # Ensure SUMO_HOME is set for the script to run
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    evaluate()