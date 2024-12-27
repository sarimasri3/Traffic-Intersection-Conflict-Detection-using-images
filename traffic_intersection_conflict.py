# -*- coding: utf-8 -*-
"""
Title: Traffic Intersection Conflict Detection & Fine-Tuning

Description:
  This script demonstrates how to:
    1. Read a labeled dataset of 4-way intersection scenarios.
    2. Balance the dataset based on conflict = yes/no.
    3. Save the balanced dataset to a new CSV file.
    4. Optionally fine-tune an OpenAI model on these data samples.
    5. Evaluate the fine-tuned model (or do a zero-shot evaluation) on a test set.
    6. Provide utility functions for reading images, encoding them, sending them to the model, etc.

Author: Sari Masri
Date: 2024-12-27
"""

import os
import time
import base64
import mimetypes
import json
import re

# Google Colab integration (optional: remove if not using Colab)
# from google.colab import drive

import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Image handling
from PIL import Image

# OpenAI libraries
from openai import OpenAI
client = OpenAI(api_key='openai-key')

# -------------------------------------------------------------------------
# Section A: Basic Data Balancing
# -------------------------------------------------------------------------
def balance_conflict_data(input_csv_path, output_csv_path):
    """
    1) Load a dataset from `input_csv_path`.
    2) Filter for '4 ways' with vehicles > 0.
    3) Balance conflict = yes/no (downsample larger class).
    4) Save the balanced DataFrame to `output_csv_path`.
    5) Return the balanced DataFrame.
    """
    df = pd.read_csv(input_csv_path)

    # Filter for 4-way intersections with vehicles > 0
    df_4ways = df[
        (df['intersection_layout'] == '4 ways') &
        (df['vehicles'] > 0)
    ]

    # Separate data based on conflict
    conflict_yes = df_4ways[df_4ways['conflict'] == 'yes']
    conflict_no = df_4ways[df_4ways['conflict'] == 'no']

    # Determine the smaller group size
    min_size = min(len(conflict_yes), len(conflict_no))

    # Sample from both groups to balance the dataset
    balanced_yes = conflict_yes.sample(n=min_size, random_state=42)
    balanced_no = conflict_no.sample(n=min_size, random_state=42)

    # Combine the balanced samples
    balanced_df = pd.concat([balanced_yes, balanced_no])
    balanced_df.to_csv(output_csv_path, index=False)

    print("Balanced dataset created and saved to:", output_csv_path)
    print(balanced_df['conflict'].value_counts())
    return balanced_df

# -------------------------------------------------------------------------
# Section B: Image Handling & GPT Request (Example)
# -------------------------------------------------------------------------
def list_files_in_folder(folder_path):
    """List all image files in the specified folder."""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(supported_formats)
    ])
    return files

def read_image_file(image_path):
    """Read an image file and return its content and MIME type."""
    with open(image_path, 'rb') as f:
        image_content = f.read()
    mime_type, _ = mimetypes.guess_type(image_path)
    return image_content, mime_type

def encode_image_to_base64(image_content):
    """Encode image content to a base64 string."""
    return base64.b64encode(image_content).decode('utf-8')

def analyze_images_in_order(folder_path, openai_api_key):
    """
    Example function to send a prompt plus multiple images to GPT-4 (or a fine-tuned model).

    Args:
        folder_path (str): Path containing sequential frames (e.g., frame_1.jpg, frame_2.jpg, etc.).
        openai_api_key (str): Your OpenAI API key.
    """

    images = list_files_in_folder(folder_path)
    if not images:
        print('No images found in the specified folder:', folder_path)
        return

    # Example system prompt
    system_prompt = """You are an advanced traffic control AI analyzing drone footage of a four-way intersection.
Output "yes" if conflict, otherwise "no"."""

    # Build the message content
    messages = [{"role": "system", "content": system_prompt}]

    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        image_content, mime_type = read_image_file(image_path)
        img_b64_str = encode_image_to_base64(image_content)

        # Append the image data as a user message
        messages.append(
            {
                "role": "user",
                "content": f"data:{mime_type};base64,{img_b64_str}"
            }
        )

    # Send the request to GPT
    response = client.chat.completions.create(
        model="gpt-4",  # or your fine-tuned model
        messages=messages
    )
    analysis = response.choices[0].message.content
    print(f"Analysis:\n{analysis}\n")

# -------------------------------------------------------------------------
# Section C: Fine-Tuning Data Preparation
# -------------------------------------------------------------------------
def extract_frame_urls(scenario_images_path):
    """
    Extract URLs for frame_3, frame_4, and frame_5 from a scenario_images_path column
    that is stored like a Python dictionary in string form.
    """
    # Convert string dictionary to Python dict (handling single quotes)
    frame_data = json.loads(scenario_images_path.replace("'", '"'))
    # Return the URLs for frames 3, 4, 5
    return [frame_data.get(f"frame_{i}.jpg") for i in range(3, 6)]

def create_finetune_example(row, system_prompt):
    """
    Create a single training example in the JSONL format for OpenAI ChatCompletion.

    row: must contain 'scenario_images_path' and 'conflict'.
    system_prompt: the instructions provided to the system (role="system").
    """
    image_urls = extract_frame_urls(row['scenario_images_path'])

    # Construct user messages for each image
    image_messages = []
    for url in image_urls:
        if url:
            image_messages.append({"role": "user", "content": f"Image URL: {url}"})

    # The user prompt
    user_prompt = "Below are three frames from the intersection. Determine if there's a conflict (yes/no)."

    # Combine into message sequence
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    messages.extend(image_messages)
    # Provide the correct label from the dataset
    messages.append({"role": "assistant", "content": row['conflict'].strip().lower()})

    return messages

def write_jsonl(df, filename, system_prompt):
    """
    Writes a DataFrame to a JSONL file, line by line, suitable for OpenAI fine-tuning.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Each example is a dictionary with a "messages" key
            messages = create_finetune_example(row, system_prompt)
            json_line = json.dumps({"messages": messages})
            f.write(json_line + "\n")

# -------------------------------------------------------------------------
# Section D: Fine-Tuning Workflow Example
# -------------------------------------------------------------------------
def fine_tune_example():
    """
    1. Load a balanced dataset
    2. Split into train/val/test
    3. Create JSONL files
    4. Upload & create fine-tuning job
    5. Poll for completion
    """
    # Example dataset path; adjust as needed
    dataset_path = "/content/drive/MyDrive/Traffic intersections workspace/frames_dataset/scenarios/four_ways_balanced_labeled_updated_direct_image_urls_dataset.csv"
    df = pd.read_csv(dataset_path)
    print("Dataset loaded. Conflict distribution:")
    print(df['conflict'].value_counts())

    # Split train/val/test
    df_train_full, df_test = train_test_split(df, test_size=0.2, stratify=df['conflict'], random_state=42)
    df_finetune_train, df_finetune_val = train_test_split(df_train_full, test_size=0.1, stratify=df_train_full['conflict'], random_state=42)

    # Create JSONL files
    system_prompt = (
        "Analyze three sequential overhead images of a four-leg intersection... "
        "Answer strictly 'yes' or 'no' for conflict."
    )
    write_jsonl(df_finetune_train, "finetune_train.jsonl", system_prompt)
    write_jsonl(df_finetune_val, "finetune_val.jsonl", system_prompt)


    train_response = client.files.create(file=open("finetune_train.jsonl", "rb"), purpose="fine-tune")
    val_response  = client.files.create(file=open("finetune_val.jsonl",  "rb"), purpose="fine-tune")

    # Create fine-tuning job
    fine_tune_job = client.fine_tuning.jobs.create(
        model="gpt-3.5-turbo",  # or your desired model
        training_file=train_response.id,
        validation_file=val_response.id
    )
    print("Fine-tune job created:", fine_tune_job)

    # Polling
    ft_id = fine_tune_job["id"]
    while True:
        status_resp = client.fine_tuning.jobs.retrieve(id=ft_id)
        status = status_resp["status"]
        if status == "succeeded":
            print("Fine-tuning succeeded:", status_resp)
            break
        elif status == "failed":
            print("Fine-tuning failed:", status_resp)
            break
        else:
            print(f"Fine-tuning status: {status}. Waiting 60s...")
            time.sleep(60)

# -------------------------------------------------------------------------
# Section E: Zero-Shot Evaluation
# -------------------------------------------------------------------------
def analyze_images_with_urls_zero_shot(image_urls):
    """
    Placeholder function for zero-shot evaluation. 
    Currently returns "no" for all scenarios.
    Replace or expand with your actual model logic.
    """
    # Example logic:
    # response = client.chat.completions.create(...)
    # predicted_label = response.choices[0].message.content.strip().lower()
    return "no"

def evaluate_zero_shot(df):
    """
    Evaluate zero-shot performance on a DataFrame with 'scenario_images_path' & 'conflict'.
    """
    y_true, y_pred = [], []

    for i, row in df.iterrows():
        scenario_images_path = row['scenario_images_path']
        true_label = row['conflict'].strip().lower()
        image_urls = extract_frame_urls(scenario_images_path)

        predicted_label = analyze_images_with_urls_zero_shot(image_urls)
        if predicted_label not in ["yes", "no"]:
            predicted_label = "no"
        
        y_true.append(true_label)
        y_pred.append(predicted_label)
        print(f"Row {i+1}: True={true_label}, Pred={predicted_label}")

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    print("Zero-Shot Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=["yes", "no"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["yes", "no"],
                yticklabels=["yes", "no"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Zero-Shot Confusion Matrix")
    plt.show()

# -------------------------------------------------------------------------
# Section F: Evaluate with Fine-Tuned Model
# -------------------------------------------------------------------------
def evaluate_with_fine_tuned_model(df, fine_tuned_model_id, max_retries=3, retry_delay=10):
    """
    Evaluate a fine-tuned model on the DataFrame with 'scenario_images_path' and 'conflict'.
    Attempts up to `max_retries` for each scenario if there's a request error.
    """
    y_true, y_pred = [], []
    total_samples = len(df)

    system_prompt = (
        "Analyze three sequential overhead images of a four-leg intersection, 0.5s apart. "
        "Answer strictly 'yes' or 'no' in lowercase to detect conflicts."
    )
    user_prompt = "Below are three frames from the intersection."

    for index, row in df.iterrows():
        true_label = row['conflict'].strip().lower()
        image_urls = extract_frame_urls(row['scenario_images_path'])

        # Build the message list
        image_messages = []
        for url in image_urls:
            if url:
                image_messages.append({"role": "user", "content": f"Image: {url}"})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        messages.extend(image_messages)

        # Attempt request with retries
        predicted_label = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=fine_tuned_model_id,
                    messages=messages
                )
                predicted_label = resp.choices[0].message.content.strip().lower()
                break
            except Exception as e:
                print(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(retry_delay)

        if predicted_label not in ["yes", "no"]:
            predicted_label = "no"

        y_true.append(true_label)
        y_pred.append(predicted_label)
        print(f"Scenario {index+1}/{total_samples}: True={true_label}, Pred={predicted_label}")

    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=["yes", "no"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["yes", "no"],
                yticklabels=["yes", "no"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Fine-Tuned Model Confusion Matrix")
    plt.show()

# -------------------------------------------------------------------------
# Section G: Additional Evaluation Function (with Explanation to CSV)
# -------------------------------------------------------------------------
def evaluate_dataset(
    df,
    fine_tuned_model_id,
    max_retries=3,
    retry_delay=10,
    output_csv_path="evaluation_results.csv"
):
    """
    Evaluate the fine-tuned model on the given DataFrame, saving results (including explanations)
    to a CSV file. Returns a DataFrame containing the evaluation details.
    """

    fieldnames = [
        "scenario_id", "scenario_path", "scenario_images_path",
        "actual_conflict", "predicted_conflict",
        "explanation", "actions"
    ]

    # If you have columns named scenario_id or scenario_path, they will be used; otherwise blank
    y_true, y_pred = [], []
    total_samples = len(df)
    count = 1

    system_prompt = """Analyze three sequential overhead images of a four-leg intersection, 0.5s apart..."""
    user_prompt = """Below are three frames from the intersection. Determine conflict (yes/no) and explain reasoning.
Use the following format:
answer: yes or no
explanation: <details>
actions: <list of vehicles>"""

    file_empty = True
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        file_empty = False

    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = pd.io.common.Dialect.delimiter  # just for placeholder
        writer = None  # reset
        import csv
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if file_empty:
            writer.writeheader()

        for _, row in df.iterrows():
            scenario_images_path = row.get('scenario_images_path', '')
            true_label = row.get('conflict', '').strip().lower()
            scenario_path = row.get('scenario_path', '')
            scenario_id = row.get('scenario_id', '')

            image_urls = extract_frame_urls(scenario_images_path)
            image_messages = []
            for url in image_urls:
                if url:
                    image_messages.append({"role": "user", "content": f"Image URL: {url}"})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            messages.extend(image_messages)

            response_text = None
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=fine_tuned_model_id,
                        messages=messages
                    )
                    response_text = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    print(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(retry_delay)

            if not response_text:
                # default if no response
                response_text = "answer: no\nexplanation: No valid response.\nactions: N/A"

            # Parse the answer, explanation, actions
            answer_match = re.search(r"(?i)answer:\s*(yes|no)", response_text)
            explanation_match = re.search(r"(?i)explanation:\s*(.*)", response_text)
            actions_match = re.search(r"(?i)actions:\s*(.*)", response_text)

            if answer_match:
                predicted_label = answer_match.group(1).lower()
            else:
                predicted_label = "no"

            explanation = explanation_match.group(1).strip() if explanation_match else ""
            actions = actions_match.group(1).strip() if actions_match else ""

            y_true.append(true_label)
            y_pred.append(predicted_label)

            writer.writerow({
                "scenario_id": scenario_id,
                "scenario_path": scenario_path,
                "scenario_images_path": scenario_images_path,
                "actual_conflict": true_label,
                "predicted_conflict": predicted_label,
                "explanation": explanation,
                "actions": actions,
            })

            print(f"Scenario {count}/{total_samples}: True: {true_label}, Pred: {predicted_label}")
            count += 1

    # Now compute overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=["yes", "no"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["yes", "no"], yticklabels=["yes", "no"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    results_df = pd.read_csv(output_csv_path)
    return results_df

# -------------------------------------------------------------------------
# Section H: Main / Example Usage
# -------------------------------------------------------------------------
if __name__ == "__main__":

    # Example: Balancing a dataset
    # balanced_df = balance_conflict_data(
    #     input_csv_path="/content/drive/MyDrive/Traffic intersections workspace/frames_dataset/scenarios/dataset_labeled.csv",
    #     output_csv_path="/content/drive/MyDrive/Traffic intersections workspace/frames_dataset/scenarios/balanced_dataset.csv"
    # )

    # Example: Analyzing images from a local folder
    # analyze_images_in_order(
    #     folder_path="/content/drive/MyDrive/Traffic intersections workspace/frames_dataset/scenarios/4_ways_scenarios/scenario_2711",
    #     openai_api_key="YOUR_OPENAI_API_KEY"
    # )

    # Example: Fine-tuning flow
    # fine_tune_example()

    # Example: Evaluating zero-shot with a test set
    # df_test = pd.read_csv("/content/drive/MyDrive/Traffic intersections workspace/frames_dataset/scenarios/test_dataset2.csv")
    # evaluate_zero_shot(df_test)

    # Example: Evaluating with a fine-tuned model
    # fine_tuned_model_id = "ft:gpt-4-xxxx-xxxx"  # your actual fine-tuned model ID
    # evaluate_with_fine_tuned_model(df_test, fine_tuned_model_id)

    # Example: Evaluate dataset with explanations, saving to CSV
    # results_df = evaluate_dataset(
    #     df_test, 
    #     fine_tuned_model_id="ft:gpt-4-xxxx-xxxx", 
    #     output_csv_path="/content/drive/MyDrive/Traffic intersections workspace/frames_dataset/scenarios/evaluation_results_3.csv"
    # )
    # print(results_df.head())

    print("Script loaded. Uncomment the relevant sections in __main__ to run them.")
