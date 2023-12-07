import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

def map_acceptability(acceptability):
    if acceptability:
        if re.search(r'accept.*', acceptability, re.IGNORECASE):
            return 1
        else:
            return 0
    return -1  # Default value if acceptability column is not found

def map_difficulty(difficulty):
    difficulty = difficulty.lower()
    if any(substring in difficulty for substring in ['impossible', 'impractical']):
        return 0
    elif 'conceptually impossible' in difficulty:
        return 1
    elif 'challenging' in difficulty:
        return 2
    elif 'feasible' in difficulty:
        return 3
    elif 'easy' in difficulty:
        return 4
    else:
        return -1  # Default value if difficulty doesn't match any predefined categories


def parse_responses(file_name):
    with open(file_name, 'r') as file:
        file_content = file.read()

    lines = file_content.strip().split('\n')
    data = []
    headers = ['Task', 'Acceptability', 'Task Difficulty', 'Explanation']
    model_url = re.findall(r'(https?://\S+)', file_content)  # Extract URLs
    model_name = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b \(?(https?://\S+)?\)?', file_content)  # Extract Model Names and URLs

    for line in lines[2:]:
        parts = re.split(r'\t+|\s{2,}', line.strip())
        task = parts[0]

        acceptability = -1
        if len(parts) > 1:
            acceptability = map_acceptability(parts[1])

        task_difficulty = -1
        if len(parts) > 2:
            task_difficulty = map_difficulty(parts[2])

        explanation = ' '.join(parts[3:])

        data.append({
            headers[0]: task,
            headers[1]: acceptability,
            headers[2]: task_difficulty,
            headers[3]: explanation,
            'Model URL': model_url[0] if model_url else None,
            'Model Name': model_name[0][0] if model_name else None,
            'Filename': file_name  # Include Filename in DataFrame
        })

    return pd.DataFrame(data, columns=headers)

def calculate_statistics(data):
    stats = data.groupby(['Filename', 'Configuration', 'Model Name', 'Model URL', 'Acceptability', 'Task Difficulty']).size().reset_index(name='Count')
    agg_stats = data.groupby(['Filename', 'Configuration', 'Model Name', 'Model URL']).agg({
        'Acceptability': ['count', 'min', 'max', 'median', lambda x: x.mode().iloc[0] if not x.mode().empty else None],
        'Task Difficulty': ['min', 'max', 'median', lambda x: x.mode().iloc[0] if not x.mode().empty else None]
    }).reset_index()

    return stats, agg_stats

def visualize_data(data):
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    sns.countplot(data=data, x='Acceptability', hue='Task Difficulty')
    plt.xlabel('Acceptability')
    plt.ylabel('Count')
    plt.title('Task Acceptability and Difficulty')
    plt.legend(title='Task Difficulty')

    plt.subplot(2, 2, 2)
    sns.countplot(data=data, x='Model', hue='Acceptability')
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.title('Acceptability by Model')
    plt.legend(title='Acceptability')

    plt.subplot(2, 2, 3)
    sns.countplot(data=data, x='Configuration', hue='Acceptability')
    plt.xlabel('Configuration')
    plt.ylabel('Count')
    plt.title('Acceptability by Configuration')
    plt.legend(title='Acceptability')

    plt.subplot(2, 2, 4)
    sns.countplot(data=data, x='Task Difficulty')
    plt.xlabel('Task Difficulty')
    plt.ylabel('Count')
    plt.title('Task Difficulty Distribution')

    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process folder of files')
    parser.add_argument('--folder_path', type=str, default='responses', help='Path to the folder containing files')
    parser.add_argument('--output_csv', type=str, default='default_output.csv', help='Path to output CSV file')
    parser.add_argument('--output_pdf', type=str, default='default_output.pdf', help='Path to output PDF file')
    parser.add_argument('--statistics_csv', type=str, default='statistics_output.csv', help='Path to output statistics CSV file')
    args = parser.parse_args()

    files = [os.path.join(args.folder_path, file_name) for file_name in os.listdir(args.folder_path)
             if file_name.endswith('.txt') or file_name.endswith('.md')]

    parsed_data = []
    for file_name in files:
        parsed_data.append(parse_responses(file_name))
    df = pd.concat(parsed_data, ignore_index=True)

    statistics = calculate_statistics(df)

    visualize_data(df)
    plt.savefig(args.output_pdf)
    plt.close()

    df.to_csv(args.output_csv, index=False)
    statistics.to_csv(args.statistics_csv, index=False)