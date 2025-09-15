import os
import sys
import traci
from sumolib import checkBinary

# --- Helper function for SUMO_HOME check ---
def ensure_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

def run_fixed_simulation():
    """Main simulation loop for a FIXED-TIMER traffic light."""

    SUMO_BINARY = checkBinary('sumo')
    CONFIG_FILE = "simulation/config.sumocfg"
    TRAFFIC_LIGHT_IDS = ["J1", "J2"]
    MAX_STEPS = 3600

    # We run for ONE episode to get the baseline performance
    # Output is saved to a separate file to avoid overwriting AI results
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--tripinfo-output", "tripinfo_fixed.xml"])

    step = 0
    phase_timer = 0
    PHASE_DURATION = 30 # Switch light every 30 steps (seconds)

    print("Running fixed-timer baseline simulation...")
    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1
        phase_timer += 1

        # If the timer exceeds the duration, switch all lights and reset
        if phase_timer > PHASE_DURATION:
            for tl_id in TRAFFIC_LIGHT_IDS:
                current_phase = traci.trafficlight.getPhase(tl_id)
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
                num_phases = len(logic[0].phases)
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(tl_id, next_phase)
            phase_timer = 0 # Reset the timer

    traci.close()
    print("Fixed-timer simulation finished. Results saved to 'tripinfo_fixed.xml'.")

if __name__ == "__main__":
    ensure_sumo_home()
    run_fixed_simulation()