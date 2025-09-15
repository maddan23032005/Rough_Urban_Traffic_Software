import xml.etree.ElementTree as ET
import numpy as np

def analyze_tripinfo(file_path):
    """
    Parses a SUMO tripinfo XML file and returns key performance metrics.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    durations = []
    wait_times = []

    for tripinfo in root.findall('tripinfo'):
        durations.append(float(tripinfo.get('duration')))
        wait_times.append(float(tripinfo.get('waitingTime')))

    if not durations:
        print(f"Warning: No trip data found in {file_path}.")
        return None, None

    avg_duration = np.mean(durations)
    avg_wait_time = np.mean(wait_times)

    print(f"\n--- Analysis for: {file_path} ---")
    print(f"Average Trip Duration: {avg_duration:.2f} seconds")
    print(f"Average Waiting Time:  {avg_wait_time:.2f} seconds")
    print("------------------------------------")

    return avg_duration, avg_wait_time

if __name__ == "__main__":
    # Analyze the two files
    ai_duration, _ = analyze_tripinfo("tripinfo_trained_ai.xml")
    fixed_duration, _ = analyze_tripinfo("tripinfo_fixed.xml")

    # Compare the results
    if ai_duration is not None and fixed_duration is not None:
        print("\n\n--- FINAL COMPARISON ---")
        improvement = ((fixed_duration - ai_duration) / fixed_duration) * 100
        print(f"Baseline Average Commute Time (Fixed-Timer): {fixed_duration:.2f} s")
        print(f"AI Model Average Commute Time:                {ai_duration:.2f} s")
        print("\n------------------------------------")
        if improvement > 0:
            print(f"🏆 Your AI improved the average commute time by {improvement:.2f}%")
        else:
            print(f"Your AI resulted in a {abs(improvement):.2f}% longer commute time.")