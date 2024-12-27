# Traffic Intersection Conflict Detection & Fine-Tuning

This repository provides a Python script to detect potential traffic conflicts at four-way intersections using labeled data and overhead images (frames). It supports:

1. Balancing datasets for conflict detection (conflict = yes/no).
2. Fine-tuning OpenAI models on labeled traffic data.
3. Evaluating both fine-tuned and zero-shot models for performance.
4. Processing image data for Base64 encoding and integration with models.

---

## Installation

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2. Install dependencies
Use the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

Or install them manually:
```bash
pip install openai httpx pandas scikit-learn matplotlib seaborn pillow
```

### 3. Set your OpenAI API key
You can set the API key by either:
- **Editing the script**: Replace `YOUR_OPENAI_API_KEY` with your actual API key in `traffic_intersection_conflict.py`.
- **Setting an environment variable**:
```bash
export OPENAI_API_KEY="sk-..."
```

### 4. Adjust file paths
Update file paths in `traffic_intersection_conflict.py` to match your local setup if you're not using Google Colab.

---

## Usage

### 1. Balance your dataset
Uncomment the following lines in the `__main__` block of the script:
```python
balanced_df = balance_conflict_data(
    input_csv_path="path_to_your_unbalanced_dataset.csv",
    output_csv_path="balanced_dataset.csv"
)
```

### 2. Fine-tune the model
- Ensure your dataset is properly labeled and train/val/test splits are ready.
- Uncomment the relevant lines:
```python
fine_tune_example()
```
This generates `finetune_train.jsonl` and `finetune_val.jsonl`, then starts a fine-tuning job with OpenAI.

### 3. Evaluate the model
#### Zero-shot evaluation:
```python
evaluate_zero_shot(df_test)
```

#### Fine-tuned model evaluation:
```python
evaluate_with_fine_tuned_model(df_test, fine_tuned_model_id="ft:gpt-4-xxxxxx")
```

#### Evaluation with explanations saved to CSV:
```python
results_df = evaluate_dataset(
    df_test,
    fine_tuned_model_id="ft:gpt-4-xxxxxx",
    output_csv_path="evaluation_results.csv"
)
```

### 4. Run the script
To execute the script:
```bash
python traffic_intersection_conflict.py
```
Uncomment the sections you need in the `if __name__ == "__main__":` block.

---

## Notes

- **Colab Integration**: If running in Colab, retain `drive.mount` statements and use paths like `/content/drive/MyDrive/...`.
- **Local Usage**: Ensure your CSV and image directories are accessible from your local machine.
- **Image Data**: This script encodes images into Base64 for API integration. Verify compliance with OpenAI's size and format limits.
- **API Key Security**: Do not commit your API key to a public repository. Use environment variables instead.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Submit a pull request or open an issue to discuss potential changes or fixes.

---

## Contact / Support

For questions, support, or issues, please open an issue on GitHub or email: [sarimasri3@gmail.com](mailto:sarimasri3@gmail.com).
[huthaifa.ashqar@aaup.edu](mailto:huthaifa.ashqar@aaup.edu).

---

