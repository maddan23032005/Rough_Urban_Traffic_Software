import os
import sys
import traci
from sumolib import checkBinary
import numpy as np
from agent.dqn_agent import DQNAgent

# --- SUMO Configuration ---
SUMO_BINARY = checkBinary('sumo') # Using non-GUI for faster training
CONFIG_FILE = "simulation/config.sumocfg"

# --- Agent and Simulation Parameters ---
EPISODES = 50 # Set to 50 for full training
MAX_STEPS_PER_EPISODE = 3600
TRAFFIC_LIGHT_IDS = ["J1", "J2"]
NUM_AGENTS = len(TRAFFIC_LIGHT_IDS)

# Define state and action spaces
STATE_SIZE = 4 
ACTION_SIZE = 2 

def get_state(traffic_light_id):
    """Retrieves the state for a traffic light agent."""
    lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
    state = [
        max(queue_lengths[0], queue_lengths[2]), # Simplified N-S
        max(queue_lengths[1], queue_lengths[3])  # Simplified E-W
    ]
    while len(state) < STATE_SIZE:
        state.append(0)
    return np.array(state[:STATE_SIZE])


def run_simulation():
    """Main simulation and training loop."""
    agents = {tl_id: DQNAgent(STATE_SIZE, ACTION_SIZE, tl_id) for tl_id in TRAFFIC_LIGHT_IDS}
    
    for e in range(EPISODES):
        traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--tripinfo-output", "tripinfo.xml"])
        
        step = 0
        total_episode_reward = {tl_id: 0 for tl_id in TRAFFIC_LIGHT_IDS}
        current_states = {tl_id: get_state(tl_id) for tl_id in TRAFFIC_LIGHT_IDS}
        
        while step < MAX_STEPS_PER_EPISODE:
            actions = {tl_id: agents[tl_id].act(current_states[tl_id]) for tl_id in TRAFFIC_LIGHT_IDS}

            for tl_id, action in actions.items():
                if action == 1:
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
                    num_phases = len(logic[0].phases)
                    next_phase = (current_phase + 1) % num_phases
                    traci.trafficlight.setPhase(tl_id, next_phase)

            for _ in range(10):
                traci.simulationStep()
                step += 1

            next_states = {tl_id: get_state(tl_id) for tl_id in TRAFFIC_LIGHT_IDS}
            
            total_system_wait_time = sum(np.sum(s) for s in next_states.values())
            global_reward = -total_system_wait_time

            for tl_id in TRAFFIC_LIGHT_IDS:
                agent = agents[tl_id]
                reward = global_reward
                done = step >= MAX_STEPS_PER_EPISODE
                agent.remember(current_states[tl_id], actions[tl_id], reward, next_states[tl_id], done)
                total_episode_reward[tl_id] += reward

            current_states = next_states

            for agent in agents.values():
                agent.replay()
                
            if done:
                break
        
        print(f"Episode {e+1}/{EPISODES} - Rewards: {total_episode_reward}")
        traci.close()

    # --- RETURN THE TRAINED AGENTS AFTER THE LOOP ---
    return agents

if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
        
    # --- RUN TRAINING AND SAVE THE MODELS ---
    trained_agents = run_simulation()
    
    print("\nTraining finished. Saving models...")
    for tl_id, agent in trained_agents.items():
        agent.save(f"{tl_id}_model.pth")
    print("Models saved successfully to J1_model.pth and J2_model.pth")