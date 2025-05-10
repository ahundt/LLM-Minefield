# Final Code: minefield_run.py - v4 (This is the complete code ready to use)
# Incorporates direct calls to minefield_summary functions for analysis

"""
Script to run multiple LLM models on prompts extracted from input files,
save the combined outputs in a timestamped run folder, and then
execute minefield_summary.py's analysis logic by calling its functions.

Arguments and console output are logged to the unique run folder.

Requirements:
- Python 3.8+
- 'ollama' Python package installed (`pip install ollama`)
- An Ollama server running (default: http://localhost:11434)
- The models specified via --models pulled in Ollama (e.g., `ollama pull llama3.2`)
- minefield_summary.py located in the same directory as minefield_run.py
- **All dependencies required by minefield_summary.py** (e.g., pandas, numpy, seaborn, matplotlib, plotly) must be installed in the **same Python environment** running minefield_run.py.
- Input prompt files (.txt or .md) located in the --input_folder (default: 'responses')
  - These files should have the prompt content first, optionally followed by
    sections formatted like "Model Name (Model URL):" for existing outputs.
    The script extracts everything before the first such section, reusing logic
    from minefield_summary.py.
"""

import argparse
import os
import datetime
import re
import json
import sys # Added for console logging
# builtins is useful for debugging if needed, but sys is sufficient for redirection
import builtins # Added for console logging (optional, good for print before redirection)
import traceback # Added for logging tracebacks

# Import the summary script as requested.
# We now call its functions directly for analysis.
import minefield_summary # type: ignore # Type ignore for static analysis if summary isn't a standard package
from tqdm import tqdm # Import tqdm for progress bars
# Import ollama for LLM interaction
# Note: Requires 'pip install ollama'
try:
    import ollama
except ImportError:
    # Print directly to original stderr in case logging fails later
    builtins.print("Error: The 'ollama' library is not installed.", file=sys.stderr)
    builtins.print("Please install it using: pip install ollama", file=sys.stderr)
    exit(1)

# MODEL_SECTION_START_PATTERN is primarily for documentation/understanding
# as the actual splitting logic is reused from minefield_summary.
MODEL_SECTION_START_PATTERN = r"\n([^\n]+) \((https?://\S+)\):\s*\n"


def parser_setup(parser=None):
    """
    Parses command-line arguments for minefield_run.py.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description='Process folder of files')

    parser = argparse.ArgumentParser(
        description='Run multiple LLM models on prompts, store results, and analyze with minefield_summary.'
    )

    parser.add_argument(
        '--input_folder',
        type=str,
        default='responses',
        help='Path to the folder containing original prompt files (.txt or .md).'
    )

    parser.add_argument(
        '--output_base_folder',
        type=str,
        default='runs',
        help='Base path where timestamped run folders will be created (e.g., runs/run_YYYY_MM_DD_HH_MM_SS).'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+', # Expect one or more model IDs
        default=['qwen3:30b', 'gemma3:27b', 'granite3.3:latest', 'llama3.3:latest', 'phi4:latest'], # Example default model
        help='List of LLM model IDs to run (e.g., "ollama/llama3.2 ollama/mistral"). Note: requires Ollama to be running and models pulled.'
    )

    # Pass-through arguments for minefield_summary.py's process_data
    # Keep default behavior consistent with minefield_summary.py defaults
    parser.add_argument(
        '--skip_descriptor_drop',
        action='store_true', # Store True if flag is present
        help='(Passed to minefield_summary.process_data) Skip dropping specific descriptor rows.'
    )

    # The rename_models argument is parsed as JSON directly.
    # The result (a dict) is stored in args.rename_models.
    parser.add_argument(
        '--rename_models',
        type=json.loads, # Use json.loads as the type directly
        # default='{}', # Default is empty JSON string, which json.loads parses to {}
        default='{"qwen3:30b": "Qwen 3 30B Alibaba", "gemma3:27b": "Gemma 3 27B Google", "granite3.3:latest": "Granite 3.3 8B (IBM)", "llama3.3:latest": "LLaMA 3.3 27B (Meta)", "phi4:latest": "Phi 4 14B Microsoft"}',
        help='(Passed to minefield_summary.process_data) JSON string for renaming models (e.g., \'{"Bing": "CoPilot"}\').'
    )

    parser.add_argument('--resume', action='store_false', dest='resume', help='Disable resume option that is on by default. If resume is enabled, the script will skip files that have already been processed and have a log file in the logs_chat_folder. If resume is disabled, the script will overwrite existing log files.')
    parser.add_argument('--output_folder', type=str, default='', help='Path to the output folder, if blank will be loaded from the latest entry in the output_base_folder')

    # args = parser.parse_args()

    # Note: args.rename_models now holds the parsed dictionary.

    return parser


def read_prompt_markdown_file(filepath):
    """
    Reads a markdown file and extracts the prompt (first chunk) and model responses (other chunks) as raw text chunks, plus associated URLs.

    Args:
        filepath (str): Path to the markdown file.

    Returns:
        tuple: A tuple containing:
            - prompt (str): The text content before the first model section.
            - model_responses (dict): A dictionary {model_id: response_text} of existing model responses.
            - model_urls (dict): A dictionary {model_id: model_url} of associated model URLs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        # Print to the current stderr (console or log file)
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return None, {}, {}

    # Reuse minefield_summary's splitting logic.
    # minefield_summary.split_per_model_chunks returns a tuple:
    # (model_chunks[1:], model_names, model_urls, model_chunks[0])
    # We need the fourth element (the first chunk) for the prompt,
    # the first two elements for model responses and URLs.
    try:
        model_chunks, model_names, model_urls, first_chunk = minefield_summary.split_per_model_chunks(content)

        # Create dictionaries for model responses and URLs
        model_responses = {model_name: chunk.strip() for model_name, chunk in zip(model_names, model_chunks)}
        model_urls_dict = {model_name: model_url for model_name, model_url in zip(model_names, model_urls)}

        # Return the prompt, model responses, and model URLs
        return first_chunk.strip(), model_responses, model_urls_dict
    except Exception as e:
        # Handle potential errors within the summary's parsing function itself
        # This might happen if the file content is malformed in a way
        # the summary script's regex or splitting logic doesn't expect.
        # Print to the current stderr (console or log file)
        print(f"Error parsing file content using minefield_summary logic for {filepath}: {e}", file=sys.stderr)
        # Returning None signals that this file could not be processed
        return None, {}, {}

def gather_prompts(input_folder):
    """
    Scans the input folder for .txt or .md files and extracts the first chunk (prompt) from each.

    Returns:
        dict: A dictionary mapping filename (basename with extension) to the extracted prompt string.
              Returns an empty dictionary if no files are found or parsed successfully.
    """
    prompts = {}
    if not os.path.isdir(input_folder):
        # Print to the current stderr (console or log file)
        print(f"Error: Input folder not found or is not a directory: {input_folder}", file=sys.stderr)
        return prompts  # Return empty dict on error

    # List files in the directory, filter for actual files ending with .txt or .md
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and (f.endswith('.txt') or f.endswith('.md'))]

    if not files:
        # Print to the current stdout (console or log file)
        print(f"Warning: No .txt or .md files found in input folder: {input_folder}")

    for filename in files:
        filepath = os.path.join(input_folder, filename)
        prompt, _, _ = read_prompt_markdown_file(filepath)  # Only extract the prompt
        if prompt is not None:
            # Use the base filename as the key
            prompts[filename] = prompt
            # Print to the current stdout (console or log file)
            print(f"Gathered prompt from '{filename}'")
        else:
            # read_prompt_markdown_file already printed an error/warning if it failed
            pass  # Skip adding this file if it failed

    print(f"Finished gathering prompts. Found {len(prompts)} valid prompts.")
    return prompts


def create_run_folders(output_base_folder, output_folder, resume, current_run_timestamp):
    """
    Determines the run folder and timestamp, creates a timestamped run folder within the output_base_folder,
    and creates 'model_output' and 'model_analysis' subfolders inside it.

    Args:
        output_base_folder (str): The base directory for runs.
        output_folder (str): The specific output folder to use, if provided.
        resume (bool): Whether to resume from the latest run folder.
        current_run_timestamp (str): The current timestamp to use for creating a new run folder.

    Returns:
        tuple: A tuple containing (run_folder, model_output_folder, model_analysis_folder, run_timestamp),
               or (None, None, None, None) if folder creation fails.
    """
    os.makedirs(output_base_folder, exist_ok=True)
    run_folder = None
    run_timestamp = None

    # Determine the run folder and timestamp
    if output_folder:
        # Use the specified output folder
        run_folder = output_folder
        match = re.search(r'run_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})', run_folder)
        run_timestamp = match.group(1) if match else None
    elif resume and not output_folder:
        # Resume from the latest folder in the base folder
        sorted_folders = sorted(
            [folder for folder in os.listdir(output_base_folder) if os.path.isdir(os.path.join(output_base_folder, folder))]
        )
        # if there are no folders, run_folder will be None
        run_folder = os.path.join(output_base_folder, sorted_folders[-1]) if sorted_folders else None
        if run_folder:
            match = re.search(r'run_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})', run_folder)
            run_timestamp = match.group(1) if match else None
        else:
            print(f"Warning: No existing run folders found in '{output_base_folder}'. Creating a new one.", file=sys.stderr)
            run_timestamp = current_run_timestamp
            run_folder = os.path.join(output_base_folder, f"run_{run_timestamp}")
    else:
        # Create a new run folder with the current timestamp
        run_timestamp = current_run_timestamp
        run_folder = os.path.join(output_base_folder, f"run_{run_timestamp}")

    if not run_folder or not run_timestamp:
        print("Error: Unable to determine or create run folder.", file=sys.stderr)
        return None, None, None, None

    # Create subfolders for model output and analysis
    model_output_folder = os.path.join(run_folder, 'model_output')
    model_analysis_folder = os.path.join(run_folder, 'model_analysis')

    try:
        os.makedirs(model_output_folder, exist_ok=True)
        os.makedirs(model_analysis_folder, exist_ok=True)
        print(f"Created run folder structure: {run_folder}")
        return run_folder, model_output_folder, model_analysis_folder, run_timestamp
    except OSError as e:
        print(f"Error creating run folders in {output_base_folder}: {e}", file=sys.stderr)
        return None, None, None, None


def run_models_on_prompts(prompts, models, output_folder, run_timestamp, existing_outputs=None):
    """
    Runs each specified model on each gathered prompt using the Ollama API,
    skipping prompts that already have results, and saves outputs incrementally.

    Args:
        prompts (dict): Dictionary of filename: prompt string.
        models (list): List of model ID strings.
        existing_outputs (dict): Nested dictionary of existing outputs {filename: {model_id: response_text}}.
        output_folder (str): Path to the folder where outputs will be saved.
        run_timestamp (str): Timestamp to prefix output filenames.

    Returns:
        dict: A nested dictionary filename: {model_id: response_text}.
              Includes model responses, error messages, or skipped notes.
              Keys correspond to files successfully processed for model runs.
              An empty dictionary indicates no models were run or all attempts failed
              or all prompts were empty.
    """
    results = existing_outputs.copy() if existing_outputs else {}  # Start with existing outputs
    total_prompts = len(prompts)
    total_models = len(models)
    total_tasks = total_prompts * total_models
    current_task = 0

    if not prompts:
        print("No prompts available to run models on.")
        return results

    print(f"Starting model runs: {total_tasks} tasks ({total_prompts} prompts x {total_models} models)")

    # Outer loop: Iterate over models
    for model_id in tqdm(models, desc="Processing Models", unit="model", file=sys.stderr):
        # Inner loop: Iterate over prompts
        for filename, prompt in tqdm(prompts.items(), desc=f"Running '{model_id}'", unit="prompt", leave=False, file=sys.stderr):
            current_task += 1
            tqdm.write(f"[{current_task}/{total_tasks}] Running '{model_id}' on '{filename}'...")

            if filename in results and model_id in results[filename]:
                tqdm.write(f"Skipping '{filename}' for model '{model_id}' (already exists).")
                continue

            if not prompt.strip():
                tqdm.write(f"Skipping model runs for '{filename}' due to empty prompt.")
                results.setdefault(filename, {})[model_id] = "NOTE: Skipped model run due to empty prompt."
                continue

            try:
                # Run the model on the prompt
                response = ollama.chat(model=model_id, messages=[{'role': 'user', 'content': prompt}])
                # Extract the content from the response dictionary
                # Check the structure based on the ollama-python README example: response['message']['content']
                response_text = response.get('message', {}).get('content', str(response))
                if not response_text:  # Handle empty content or unexpected response structure
                    # Store an explicit message if model returned empty response
                    response_text = f"Warning: Model '{model_id}' returned empty content for '{filename}'. Full response: {response}"
                results.setdefault(filename, {})[model_id] = response_text
                # save_combined_outputs({filename: prompt}, {filename: results[filename]}, output_folder, models, run_timestamp)

            except ollama.ResponseError as e:
                # Specific error for API issues (e.g., model not found, invalid model)
                error_message = f"ERROR: Ollama Response Error (Status {e.status_code}): {e.error}"
                results.setdefault(filename, {})[model_id] = error_message
                tqdm.write(f"{error_message}")

            except Exception as e:
                # Catch any other exceptions (e.g., connection errors, unexpected issues)
                error_message = f"ERROR: An unexpected error occurred: {e}"
                results.setdefault(filename, {})[model_id] = error_message
                tqdm.write(f"{error_message}")

    # After iterating through all models and prompts, the results dictionary
    # contains the results/errors/notes for every requested model for every
    # successfully gathered prompt.
    print("\nFinished all model runs.")
    return results


def load_existing_outputs(output_folder):
    """
    Loads existing combined outputs from the specified folder, including model responses and URLs.

    Args:
        output_folder (str): Path to the folder containing combined output files.

    Returns:
        tuple: A tuple containing:
            - existing_outputs (dict): A nested dictionary {filename: {model_id: response_text}}.
            - model_urls (dict): A nested dictionary {filename: {model_id: model_url}}.
    """
    existing_outputs = {}
    model_urls = {}

    if not os.path.isdir(output_folder):
        print(f"Warning: Output folder '{output_folder}' does not exist. No existing outputs loaded.")
        return existing_outputs, model_urls

    for file in os.listdir(output_folder):
        if file.endswith(".md"):
            filepath = os.path.join(output_folder, file)
            prompt, model_responses, urls = read_prompt_markdown_file(filepath)  # Extract responses and URLs
            if model_responses:
                # Use the base filename (with extension) as the key
                existing_outputs[file] = model_responses
                model_urls[file] = urls

    print(f"Loaded existing outputs for {len(existing_outputs)} files.")
    return existing_outputs, model_urls


def save_combined_outputs(original_prompts, model_results, output_folder, models_list, model_urls):
    """
    Formats and saves the original prompt, each model's response, and associated URLs into separate
    Markdown files in the specified output folder. Files are named after the original input files.
    These files serve as input for minefield_summary analysis.

    Args:
        original_prompts (dict): Dictionary of filename: original_prompt string.
                                 Only includes prompts that were successfully gathered.
        model_results (dict): Nested dictionary filename: {model_id: response_text}.
                               Contains results, error messages, or notes about skipped runs.
                               Keys correspond to files successfully gathered in step 2.
        output_folder (str): Path to the directory where combined output files will be saved.
        models_list (list): The list of model IDs that were *requested* to run (from args).
                            Used to ensure a section for every requested model exists in the
                            output file, even if the model failed or was skipped for a
                            specific prompt.
        model_urls (dict): Nested dictionary filename: {model_id: model_url}.
                           Contains the URLs associated with each model for each file.
    """
    print(f"\n--- Saving combined model outputs to '{output_folder}' ---")

    if not os.path.isdir(output_folder):
        print(f"Error: Output folder not found or is not a directory: {output_folder}", file=sys.stderr)
        return  # Cannot save if folder is missing

    if not original_prompts:
        print("No original prompts available to save outputs.")
        # Note: model_results might still contain notes if no models were specified,
        # but if there were no prompts, there are no output files to create based on prompts.
        if model_results:
            print("Warning: Model results exist, but no original prompts were gathered. No output files will be created.")
        return

    # Iterate through the prompts that were actually gathered (and are keys in model_results)
    # This ensures we only try to create output files for valid input prompts.
    for filename, original_prompt in original_prompts.items():
        # Start the output content with the original prompt
        output_content = original_prompt.strip()

        # Add a separator/newline before model outputs start
        if output_content:
            output_content += "\n\n"
        else:
            output_content += "--- Empty Prompt ---\n\n"  # Indicate if original prompt was empty

        # Iterate through the *list of requested models* to ensure all are represented
        # in the output file for this prompt. This loop runs for every prompt in original_prompts.
        for model_id in models_list:
            # Use model_id as both name and a placeholder URL for the summary script
            # Retrieve the actual URL for the model if available, otherwise use a placeholder
            model_url = model_urls.get(filename, {}).get(model_id, f"https://ollama.com/library/{model_id}")

            # Retrieve the result/note for this model and filename from model_results.
            # model_results is guaranteed to have a key for 'filename' if we are in this loop
            # (because we iterate through original_prompts keys which are used as keys in model_results).
            # .get(model_id, ...) handles cases where a model might have failed completely
            # for a specific prompt before storing an error message, or if run_models
            # didn't populate it as expected (extra safety).
            response_text = model_results.get(filename, {}).get(model_id, f"INTERNAL ERROR: Model '{model_id}' result missing for '{filename}'.")

            # Format the model section like minefield_summary expects
            # Add leading/trailing newlines around the model section start marker
            # The pattern in minefield_summary is r"\n([^\n]+) \(...\):\s*\n"
            # Adding the initial \n makes the generated format match the pattern start.
            # Adding response + \n + \n ensures content between model headers.
            output_content += f"\n{model_id} ({model_url}):\n"
            # Add model response content, stripping extra whitespace but keeping internal structure
            output_content += response_text.strip() + "\n"
            # Add an extra newline *after* the response content to separate from the next model or end of file
            output_content += "\n"

        # Determine the output filename (use original filename but ensure .md extension)
        base, ext = os.path.splitext(filename)
        # Ensure the output file has a .md extension
        output_filename = f"{base}.md"
        output_filepath = os.path.join(output_folder, output_filename)

        try:
            # Write the combined content to the new file, specifying encoding
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(output_content)
            print(f"Saved combined output for '{filename}' to '{output_filepath}'")
        except Exception as e:
            # Print to the current stderr (console or log file)
            print(f"Error writing output file {output_filepath}: {e}", file=sys.stderr)

    print("\nFinished saving all combined model outputs.")


def perform_summary_analysis(model_output_folder, model_analysis_folder, args):
    """
    Loads data from model outputs using minefield_summary.load_data,
    processes it using minefield_summary.process_data, and saves
    analysis results to the model_analysis_folder.

    Args:
        model_output_folder (str): Path to the folder containing model outputs (input for summary logic).
        model_analysis_folder (str): Path to the folder for analysis results (output for summary logic).
        args (argparse.Namespace): Parsed command-line arguments from minefield_run.py,
                                   containing pass-through args for process_data.
    """
    # This print will go to the current stdout (console or log file)
    print(f"\n--- Performing summary analysis using minefield_summary functions ---")

    # Ensure the analysis output folder exists before starting.
    os.makedirs(model_analysis_folder, exist_ok=True)

    # Check if there are files to analyze in the model output folder
    # This ensures minefield_summary.load_data doesn't run on an empty input folder
    # and prevents subsequent errors in process_data if it expects input data.
    if not os.path.isdir(model_output_folder) or not os.listdir(model_output_folder):
        print(f"Warning: No model output files found in '{model_output_folder}' for analysis. Skipping summary analysis.")
        return # Exit if no files to process

    try:
        # Call minefield_summary.load_data directly.
        # load_data expects just the input folder path.
        print(f"Loading data from '{model_output_folder}' using minefield_summary.load_data...")
        df = minefield_summary.load_data(model_output_folder)

        # Check if load_data returned a valid DataFrame
        # load_data returns None or an empty DataFrame if no data is found/parsed.
        if df is None or df.empty:
            print(f"Warning: minefield_summary.load_data returned no data from '{model_output_folder}'. Skipping analysis processing.")
            # Note: The analysis folder was already created above.
            return # Exit if no data was loaded

        print(f"Successfully loaded {len(df)} data points from {len(df['Filename'].unique())} files.")

        # Call minefield_summary.process_data directly.
        # process_data expects: df, output_folder, output_csv, skip_descriptor_drop, rename_models
        # We need to provide appropriate values for these arguments.
        # output_folder is the model_analysis_folder.
        # output_csv is the name for the main summarized CSV file.
        # skip_descriptor_drop and rename_models come from our parsed args.
        analysis_output_csv_name = 'analysis_summary.csv' # Choose a name for the main CSV output

        print(f"Processing data and generating analysis in '{model_analysis_folder}' using minefield_summary.process_data...")
        minefield_summary.process_data(
            df=df,
            output_folder=model_analysis_folder, # Directory for all analysis outputs
            output_csv=analysis_output_csv_name, # Specific name for the main CSV file
            skip_descriptor_drop=args.skip_descriptor_drop, # Pass through boolean flag
            rename_models=args.rename_models # Pass through the parsed dictionary
        )

        print("Summary analysis using minefield_summary functions finished successfully.")

    except Exception as e:
        # Catch any exception that occurs during the direct calls to summary functions
        # Print error and traceback to the redirected stderr (which is the log file)
        print(f"\nError occurred during summary analysis using minefield_summary functions: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Print traceback to log file
        print(f"Analysis folder '{model_analysis_folder}' may contain incomplete results.", file=sys.stderr)


def main():
    """
    Main function to orchestrate the minefield LLM testing and analysis process.
    Includes logging of arguments and console output.
    """

    # Store original stdout and stderr before any potential redirection attempts
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Initial message before logging setup - goes to original console
    # Using original_stdout/stderr explicitly or builtins.print for robustness
    builtins.print("Starting minefield_run process (v4)...", file=original_stdout)  # Use builtins.print for absolute certainty

    # 1. Parse arguments
    # Parse command line arguments to get configuration
    parser = parser_setup()
    args = minefield_summary.parse_args_and_config(parser=parser)

    # 2. Gather prompts from input folder
    # This might print warnings/errors about specific files to the original console
    original_prompts = gather_prompts(args.input_folder)
    if not original_prompts:
        builtins.print("No valid prompts found. Exiting.", file=original_stderr)  # Use builtins.print
        return

    # 3. Create run folders
    # Extract the current timestamp for folder creation
    current_run_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder, model_output_folder, model_analysis_folder, run_timestamp = create_run_folders(
        output_base_folder=args.output_base_folder,
        output_folder=args.output_folder,
        resume=args.resume,
        current_run_timestamp=current_run_timestamp
    )

    if not run_folder:  # Check if folder creation failed
        builtins.print("Failed to create run folders. Exiting.", file=original_stderr)  # Use builtins.print
        return

    # --- Set up logging ---
    log_filepath = os.path.join(run_folder, 'run.log')

    log_file = None  # Initialize log_file handle
    try:
        # Open log file for writing console output
        # Use 'w' mode to overwrite any previous attempts within this run
        log_file = open(log_filepath, 'w', encoding='utf-8')

        # Log confirmation and start time to the new log file (via redirected stdout)
        print(f"Run started at {current_run_timestamp} (original run timestamp: {run_timestamp})\n")

        # --- START v4 Main Logic within logging context ---

        # 4. Load existing outputs
        # Load previously processed outputs and associated model URLs
        existing_outputs, model_urls = load_existing_outputs(model_output_folder)

        # 5. Run models on gathered prompts
        # These functions and subsequent ones will print to the log file via redirection
        # Use tqdm's `file=sys.stderr` to ensure progress bars are displayed correctly
        model_results = run_models_on_prompts(
            prompts=original_prompts,
            models=args.models,
            output_folder=model_output_folder,
            run_timestamp=run_timestamp,
            existing_outputs=existing_outputs,
        )

        # 6. Save combined outputs to the model_output folder
        save_combined_outputs(
            original_prompts=original_prompts,
            model_results=model_results,
            output_folder=model_output_folder,
            models_list=args.models,
            model_urls=model_urls
        )

        # 7. Perform summary analysis by calling minefield_summary functions directly
        # This replaces the subprocess call from previous versions.
        perform_summary_analysis(
            model_output_folder=model_output_folder,
            model_analysis_folder=model_analysis_folder,
            args=args
        )

        print("\nMinefield run process finished successfully (v4).")  # This goes to the log file

    except Exception as e:
        # Catch any exception that occurs *after* logging is set up (within the try block)
        # Print error and traceback to the redirected stderr (which is the log file)
        print(f"\nFATAL EXCEPTION occurred during run: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)  # Print traceback to log file

    finally:
        # --- Restore original stdout/stderr and close log file ---
        # Ensure streams are flushed before restoring
        # This pushes any buffered output to the log file before we change where print goes.
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original streams
        # Using builtins.print to be absolutely sure the restore message goes to the original console
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Close the log file if it was successfully opened
        if log_file is not None:
            try:
                log_file.close()
            except Exception as e:
                # Print close errors to the original stderr
                builtins.print(f"Error closing log file {log_filepath}: {e}", file=original_stderr)

    # Final message after logging is restored - goes back to original console
    # This confirms the script finished, even if errors occurred before the final print in the log.
    # Using builtins.print for absolute certainty.
    builtins.print("Minefield run process completed (check log file in run folder for full output).", file=original_stdout)

if __name__ == "__main__":
    main()