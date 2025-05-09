# Minefield Summary

This Python script processes a folder of text files and markdown files, parses the responses, and generates a summary of the data in CSV format. It also provides options to drop certain rows, rename models, and visualize the data.

## Usage

You can run the script from the command line with the following syntax:

```bash
python minefield_summary.py [--input_folder INPUT_FOLDER] [--output_folder OUTPUT_FOLDER] [--output_csv OUTPUT_CSV] [--skip_descriptor_drop] [--rename_models RENAME_MODELS] [--statistics_csv STATISTICS_CSV]
```

### Arguments

- `--input_folder`: Path to the folder containing files. Default is 'responses'.
- `--output_folder`: Path to the output folder. Default is 'results'.
- `--output_csv`: Path to output CSV file that contains a table summarizing all the loaded and parsed data. Default is 'responses.csv'.
- `--skip_descriptor_drop`: If specified, the script will skip dropping rows where the Filename contains Descriptor and the prompt task name string contains a specific string.
- `--rename_models`: JSON string representing a dictionary to rename substrings of models. Default is '{"Bing": "CoPilot", "Bard": "Gemini"}'.
- `--statistics_csv`: Path to output statistics CSV file. Default is 'statistics_output.csv'.

### Example

```bash
python minefield_summary.py --input_folder data/responses --output_folder data/results --output_csv summary.csv --rename_models '{"Bing": "CoPilot", "Bard": "Gemini"}'
```

This command will process the files in the 'data/responses' folder, output the results to the 'data/results' folder, and write the summary to 'summary.csv'. It will also rename 'Bing' to 'CoPilot' and 'Bard' to 'Gemini' in the 'Model' column.

## Output

The script outputs a CSV file that summarizes the data. It also generates a statistics CSV file and visualizes the data.

## Requirements

This script requires Python 3 and the following Python libraries:

- pandas
- seaborn
- matplotlib
- argparse
- json
- os

You can install these libraries using pip:

```bash
pip install pandas seaborn matplotlib argparse json os
```

## Note

There are some records regarding a version of Microsoft's CoPilot GPT-4 rejecting the prompts. The Authors' account was part of a Microsoft-run A/B test in which an explicit GPT-4 switch appeared in December 2023. However, this was an intermittent test and the option disappeared before all experiments could be evaluated. Therefore, the case was excluded from the paper.

## License

All Rights Reserved, Copyright Andrew Hundt 2024, Author. Individuals and/or organizations (LICENSEE) who train Artificial Intelligence (AI) or Machine Learning (ML), broadly construed, on this code directly or indirectly without the explicit written authorization of the Author(s), agree to pay the greater of $1,000 per occurrance or 1% of annual revenue. LICENSEE agrees to keep all of Author(s) accounts and services intact, whenever applicable. LICENSEE agrees to grant a worldwide, perpetual, irrevocable, and sublicensable license to the Author(s) of any resulting AI system and all associated technologies and patents, broadly construed. LICENSEE agrees that this agreement superscedes all other prior and future agreements unless provided express written consent of the Author(s). LICENSEE agrees to pay all legal fees, whenever applicable.