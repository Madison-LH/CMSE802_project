''''
Module for dataset creation

This module contains functions for
extracting and labeling frames from videos 
based on manually annotated CSVs
and organizing them into folders
named after the exhibited behavior.

Author: Madison Honore
Date: March 2025
'''


import os
import glob
import re
import cv2
import pandas as pd

def process_behavior_frames(video_id, video_paths, behavior_data, output_dir):
    accumulated_time = 0
    for video_path in video_paths:
        print(f"Processing segment: {video_path} for video ID: {video_id} with accumulated time: {accumulated_time}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = frame_number / fps + accumulated_time

            for _, row in behavior_data.iterrows():
                start_time = row['Time_start']
                stop_time = row['Time_stop']
                behavior = row['Behavior']

                if behavior == "INSC":
                    continue  # Skip INSC behavior

                behavior_dir = os.path.join(output_dir, behavior)
                os.makedirs(behavior_dir, exist_ok=True)
                frame_path = os.path.join(behavior_dir, f"{video_id}_frame_{frame_number}.jpg")

                if os.path.exists(frame_path):
                    print(f"Frame {frame_number} for video ID {video_id} in behavior {behavior} already exists. Skipping.")
                    continue

                if start_time <= current_time <= stop_time:
                    try:
                        cv2.imwrite(frame_path, frame)
                        print(f"Saved frame {frame_number} for video ID {video_id} in behavior {behavior} at {current_time}")
                    except Exception as e:
                        print(f"Error writing frame {frame_number} to {frame_path}: {e}")
                    break

        accumulated_time += duration
        cap.release()

def extract_frames_by_behavior_from_folder(video_dir, csv_dir, output_dir, frame_rate=25):
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    csv_files.sort()

    for csv_path in csv_files:
        print(f"Processing CSV file: {csv_path}")
        
        try:
            event_df = pd.read_csv(csv_path, encoding='utf-8')
            print("CSV loaded successfully.")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            continue

        if 'Time' not in event_df.columns or 'Behavior type' not in event_df.columns or 'Media file name' not in event_df.columns:
            print(f"Error: Required columns missing in {csv_path}. Skipping this file.")
            continue

        event_df['Video_ID'] = event_df['Media file name'].apply(lambda x: x.split('_')[1] if pd.notnull(x) else None)
        event_df = event_df.dropna(subset=['Video_ID'])

        video_groups = event_df.groupby('Video_ID')

        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))

        def natural_key(filename):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

        video_files_sorted = sorted(video_files, key=natural_key)
        
        video_file_map = {}
        for video_file in video_files_sorted:
            filename = os.path.basename(video_file)
            try:
                video_id = filename.split('_')[1]
                video_file_map.setdefault(video_id, []).append(video_file)
            except (IndexError, ValueError) as e:
                print(f"Skipping file {video_file}: unable to parse ID.")

        for video_id, video_data in video_groups:
            video_paths = video_file_map.get(video_id, [])
            if not video_paths:
                print(f"No videos found for ID {video_id} listed in CSV.")
                continue

            print(f"Processing video ID: {video_id} from CSV {csv_path}")

            start_behaviors = video_data[video_data['Behavior type'] == 'START']
            stop_behaviors = video_data[video_data['Behavior type'] == 'STOP']
            
            start_behaviors['Event_ID'] = range(len(start_behaviors))
            stop_behaviors['Event_ID'] = range(len(stop_behaviors))
            
            merged_behaviors = pd.merge(
                start_behaviors[['Event_ID', 'Behavior', 'Time']],
                stop_behaviors[['Event_ID', 'Behavior', 'Time']],
                on=['Event_ID', 'Behavior'],
                suffixes=('_start', '_stop')
            )

            if merged_behaviors.empty:
                print(f"No matched start-stop pairs for video ID {video_id} in CSV {csv_path}.")
                continue

            behavior_groups = merged_behaviors.groupby('Behavior')
            for behavior, behavior_data in behavior_groups:
                if behavior == "INSC":
                    continue  # Skip INSC behavior
                process_behavior_frames(video_id, video_paths, behavior_data, output_dir)

    print("Frames extracted and saved based on behavior timestamps for all CSV files in the folder.")

extract_frames_by_behavior_from_folder(
    video_dir=r'E:\masked_videos', 
    csv_dir=r'E:\Manual_Scoring_Results', 
    output_dir=r'E:\Multicategory_frames'
)
