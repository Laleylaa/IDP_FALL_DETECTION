import os
import keyboard
import time
import cv2
import sys
import pandas as pd
import subprocess
import threading
import csv



def run_detection_script():
    global process
    command = "python"
    script = "yolov5\detect.py"
    weights = "best.pt"
    img_size = "640"
    confidence = "0.25"
    save_csv = "--save-csv"
    source = "0"
    cmd = [command, script, "--weights", weights, "--img", img_size, "--conf", confidence, save_csv, "--source", source]
    process = subprocess.Popen(cmd)


def analyze_results(status_event, reset_event):
    global running
    global csv_file_path
    last_checked = 0
    fall_count = 0
    normal_status_printed = False

    while running:
        try:
            #csv_file_path = os.path.join(os.path.dirname(__file__), 'yolov5', 'runs', 'detect', 'exp', 'predictions.csv')
            df = pd.read_csv(csv_file_path, header=None)

            if len(df) > last_checked:
                # Store all newly generated labels in list
                new_labels = df.iloc[last_checked:, 1]
                last_checked = len(df)
                # Check the current label 
                for label in new_labels:
                    if label == 'fall detected':
                        fall_count += 1
                        if fall_count == 150:
                            print("High-possible Fall")
                            status_event.set() # set status_event
                            fall_count = 0
                            break
                    elif not normal_status_printed:
                        print("Normal Status")
                        normal_status_printed = True

        except FileNotFoundError:  # haven't started fall detection
            print("Waiting for file...")
        except pd.errors.EmptyDataError:  # empty loaded data
            print("File is empty, waiting for data...")

        if reset_event.is_set():
            fall_count = 0
            reset_event.clear() # clear reset event  

        time.sleep(0.1)


if __name__ == "__main__":
    running = True 
    
    # Create Event objects (False as default)
    status_event = threading.Event() # check the status is normal or high-possible fall
    reset_event = threading.Event() # check the print content needs to be reset or not

    # Clear previous content in .csv file
    csv_file_path = os.path.join(os.path.dirname(__file__), 'yolov5', 'runs', 'detect', 'exp', 'predictions.csv')
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Writing an empty row to ensure the file is not entirely empty
        csv_writer.writerow([])

    # Start two threads 
    model_thread = threading.Thread(target=run_detection_script)
    model_thread.start()
    analysis_thread = threading.Thread(target=analyze_results, args=(status_event, reset_event))
    analysis_thread.start()

    try:
        # Loop to check whether High-possible Fall is triggered
        while True:
            if status_event.is_set(): # be triggered
                time.sleep(10)  # display for 10 seconds
                print("Normal Status") # reset print
                # Clear status and set reset function
                status_event.clear() 
                reset_event.set()
            
            # Shut down program
            if keyboard.is_pressed('c'):
                cv2.destroyAllWindows()
                break
            time.sleep(1)
        
        # Shut down process and threads
        if process and process.poll() is None:
            print("Process running Stopped...")
            process.terminate()    
        running = False
        print("Code Running Stopped...")
        model_thread.join()
        analysis_thread.join()
        sys.exit(0)

    except KeyboardInterrupt:
        print("Stopping...")



