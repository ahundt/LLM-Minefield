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

    # Define regex pattern for header and various delimiters
    header_pattern = r'Task\s+Acceptability\s+Task Difficulty\s+Explanation|\| Task \| Acceptability \| Task Difficulty \| Explanation \|'
    delimiter_patterns = [r'\|', r'\t']  # Patterns for pipe and tab separation

    # Find the header pattern to identify the start of the table
    match = re.search(header_pattern, file_content, re.IGNORECASE)
    if not match:
        return pd.DataFrame()  # Return empty DataFrame if header pattern is not found

    # Extract model name and URL using a specific pattern
    model_pattern = r'([A-Za-z]+\s[A-Za-z]+\s\d+\.\d+)\s\((https?://\S+)\):'
    model_match = re.search(model_pattern, file_content)
    model_name, model_url = "", ""
    if model_match:
        model_name, model_url = model_match.groups()

    # Extract the relevant data after finding the header pattern
    data_start = match.end()
    data_text = file_content[data_start:]

    # Skip comments before the table, if any
    if '#' in data_text:
        comment_start = data_text.index('#')
        data_text = data_text[comment_start + 1:]

    data = []
    headers = ['Task', 'Acceptability', 'Task Difficulty', 'Explanation']

    # Iterate through the delimiters to split the rows
    for delimiter in delimiter_patterns:
        rows = re.findall(rf'(?s)\|?(.*?)\{delimiter}', data_text)  # Extract rows with delimiter

        for row in rows:
            columns = re.split(rf'{delimiter}', row)  # Split row using the delimiter
            cleaned_columns = [col.strip() for col in columns if col.strip()]  # Clean columns

            # Ensure the row has the expected number of columns
            if len(cleaned_columns) == len(headers):
                # Map the columns to the headers and create a dictionary
                row_data = {header: value for header, value in zip(headers, cleaned_columns)}
                row_data['Filename'] = os.path.basename(file_name)  # Include Filename in DataFrame
                row_data['Model Name'] = model_name  # Include Model Name
                row_data['Model URL'] = model_url  # Include Model URL

                data.append(row_data)

    # Convert the collected data into a DataFrame
    df = pd.DataFrame(data, columns=headers + ['Model Name', 'Model URL'])
    return df


def calculate_statistics(data):
    print(f'data.columns: {data.columns}')
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
    df.to_csv(args.output_csv, index=False)

    statistics = calculate_statistics(df)

    visualize_data(df)
    plt.savefig(args.output_pdf)
    plt.close()

    statistics.to_csv(args.statistics_csv, index=False)