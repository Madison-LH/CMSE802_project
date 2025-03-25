
"""
Module for IDA

This module contains functions for
counting how many frames belong to
each behavior class according to
manually annotated timestamps stored
on CSVs.

Author: Madison Honore
Date: March 2025
"""





import os
import glob
import re
import cv2
import pandas as pd
import multiprocessing

def process_behavior_frames(video_id, video_paths, behavior_data):
    frame_counts = {behavior: 0 for behavior in behavior_data['Behavior'].unique()}
    accumulated_time = 0
    
    for video_path in video_paths:
        print(f"Processing segment: {video_path} for video ID: {video_id} with accumulated time: {accumulated_time}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"Warning: FPS read as {fps}, setting to default 25 FPS.")
            fps = 25
        
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = round(frame_number / fps + accumulated_time, 6)  # Increased precision

            for _, row in behavior_data.iterrows():
                start_time = round(row['START'], 6)
                stop_time = round(row['STOP'], 6)
                behavior = row['Behavior']

                if start_time <= current_time <= stop_time:
                    frame_counts[behavior] += 1
                    break

        accumulated_time += duration
        cap.release()
    
    return video_id, frame_counts

def convert_behavior_data(event_df):
    """Convert BORIS CSV format to start-stop pairs with floating-point precision adjustments."""
    start_behaviors = event_df[event_df['Behavior type'] == 'START'].copy()
    stop_behaviors = event_df[event_df['Behavior type'] == 'STOP'].copy()
    
    start_behaviors = start_behaviors.rename(columns={'Time': 'START'})
    stop_behaviors = stop_behaviors.rename(columns={'Time': 'STOP'})
    
    start_behaviors['Event_ID'] = range(len(start_behaviors))
    stop_behaviors['Event_ID'] = range(len(stop_behaviors))
    
    merged_behaviors = pd.merge(
        start_behaviors[['Event_ID', 'Behavior', 'START']],
        stop_behaviors[['Event_ID', 'Behavior', 'STOP']],
        on=['Event_ID', 'Behavior']
    )
    
    merged_behaviors['START'] = merged_behaviors['START'].astype(float).round(6)
    merged_behaviors['STOP'] = merged_behaviors['STOP'].astype(float).round(6)
    
    return merged_behaviors

def count_frames_by_behavior_from_folder(video_dir, csv_dir):
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    csv_files.sort()
    total_frame_counts = {}
    
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    video_file_map = {}
    for video_file in video_files:
        filename = os.path.basename(video_file)
        try:
            video_id = filename.split('_')[1]
            video_file_map.setdefault(video_id, []).append(video_file)
        except (IndexError, ValueError) as e:
            print(f"Skipping file {video_file}: unable to parse ID.")
    
    tasks = []
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

        for video_id, video_data in video_groups:
            video_paths = video_file_map.get(video_id, [])
            if not video_paths:
                print(f"No videos found for ID {video_id} listed in CSV.")
                continue

            behavior_data = convert_behavior_data(video_data)
            tasks.append((video_id, video_paths, behavior_data))
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_behavior_frames, tasks)
    
    for video_id, frame_counts in results:
        for behavior, count in frame_counts.items():
            total_frame_counts[behavior] = total_frame_counts.get(behavior, 0) + count
    
    print("\nTotal Frame Counts Across All Videos:")
    for behavior, total_count in total_frame_counts.items():
        print(f"{behavior}: {total_count}")

if __name__ == "__main__":
    count_frames_by_behavior_from_folder(
        video_dir=r'E:\masked_videos', 
        csv_dir=r'E:\Manual_Scoring_Results'
    )


