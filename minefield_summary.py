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


def parse_table_in_chunk(chunk_text, model_name, model_url):
    # Find the start of the table based on tab or pipe delimiters
    start_idx_pipe = chunk_text.find('|')
    start_idx_tab = chunk_text.find('\t')

    # Determine the starting index of the table based on the delimiter found
    start_idx = min(start_idx for start_idx in [start_idx_pipe, start_idx_tab] if start_idx != -1)

    if start_idx == -1:
        return None  # No table found in this chunk

    # Find the end of the table based on the absence of delimiters after the start index
    end_idx = min(chunk_text.rfind('|', start_idx), chunk_text.rfind('\t', start_idx))
    if end_idx == -1:
        end_idx = len(chunk_text)

    table_text = chunk_text[start_idx:end_idx].strip()  # Extracting the table portion

    # Determine the delimiter by checking for pipes or tabs
    delimiter = '|' if '|' in table_text else '\t'

    # Read the table text using Pandas read_csv
    df = pd.read_csv(pd.compat.StringIO(table_text), sep=delimiter)

    # Add Model Name, Model URL, and other necessary columns
    df['Model Name'] = model_name
    df['Model URL'] = model_url
    df['Filename'] = ''  # Placeholder for Filename column

    return df


def split_per_model_chunks(text):
    # Define regex pattern to match model name and response URL
    pattern = r'\n([A-Z][a-z]+(?: [A-Z][a-z]+)*) \(?(https?://\S+)?\)?'
    matches = re.findall(pattern, text)
    model_names, model_urls = zip(*matches)
    model_chunks = re.split(pattern, text)[2::3]
    return model_names, model_urls, model_chunks
    return model_names, model_urls, model_chunks

def parse_responses(file_name):
    with open(file_name, 'r') as file:
        file_content = file.read()

    model_names, model_urls, model_chunks = split_per_model_chunks(file_content)
    data = []
    headers = ['Task', 'Acceptability', 'Task Difficulty', 'Explanation']

    for model_name, model_url, model_chunk in zip(model_names, model_urls, model_chunks):
        if model_chunk is None:
            continue
        # print the len of each
        print(f'len(model_name): {len(model_name)} model_name: {model_name}')
        print(f'len(model_url): {len(model_url)} model_url: {model_url}')
        print(f'len(model_chunk): {len(model_chunk)}')
        parsed_table = parse_table_in_chunk(model_chunk, model_name.strip(), model_url.strip())
        if parsed_table is not None:
            parsed_table['Filename'] = os.path.basename(file_name)
            data.append(parsed_table)

    return pd.concat(data, ignore_index=True) if data else pd.DataFrame(columns=headers)



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