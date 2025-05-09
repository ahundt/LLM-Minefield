# minefield_run.py - v3
"""
Script to run multiple LLM models on prompts extracted from input files,
save the combined outputs in a timestamped run folder, and then
execute minefield_summary.py to analyze the results.

Arguments and console output are logged to the unique run folder.

Requirements:
- Python 3.8+
- 'ollama' Python package installed (`pip install ollama`)
- An Ollama server running (default: http://localhost:11434)
- The models specified via --models pulled in Ollama (e.g., `ollama pull llama3.2`)
- minefield_summary.py located in the same directory as minefield_run.py
- Input prompt files (.txt or .md) located in the --input_folder (default: 'responses')
  - These files should have the prompt content first, optionally followed by
    sections formatted like "Model Name (Model URL):" for existing outputs.
    The script extracts everything before the first such section, reusing logic
    from minefield_summary.py.
"""

import argparse
import os
import datetime
import subprocess
import re
import json
import sys # Added for console logging
import builtins # Added for console logging (optional, kept for potential debug use)
import traceback # Added for logging tracebacks

# Import the summary script as requested, although it's run via subprocess
# It is also used here for function reuse (split_per_model_chunks)
import minefield_summary # type: ignore

# Import ollama for LLM interaction
# Note: Requires 'pip install ollama'
try:
    import ollama
except ImportError:
    # Print directly to original stderr in case logging fails later
    print("Error: The 'ollama' library is not installed.", file=sys.stderr)
    print("Please install it using: pip install ollama", file=sys.stderr)
    exit(1)

# MODEL_SECTION_START_PATTERN is now primarily for documentation/understanding
# as the actual splitting logic is reused from minefield_summary.
# We keep it here as it was defined in v1, but note its reduced role.
# Define the pattern used to identify the start of a model section in input files.
# This pattern is derived from minefield_summary.split_per_model_chunks.
# Everything *before* the first match of this pattern is considered the prompt (first chunk).
# Keeping this defined here for clarity, although minefield_summary.split_per_model_chunks
# relies on its own internal pattern definition.
MODEL_SECTION_START_PATTERN = r"\n([^\n]+) \((https?://\S+)\):\s*\n"


def parse_args():
    """
    Parses command-line arguments for minefield_run.py.
    """
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
        default=['ollama/llama3.2'], # Example default model
        help='List of LLM model IDs to run (e.g., "ollama/llama3.2 ollama/mistral"). Note: requires Ollama to be running and models pulled.'
    )

    # Pass-through arguments for minefield_summary.py
    # Keep default behavior consistent with minefield_summary.py defaults
    parser.add_argument(
        '--skip_descriptor_drop',
        action='store_true', # Store True if flag is present
        help='(Passed to minefield_summary.py) Skip dropping specific descriptor rows.'
    )

    # The rename_models argument is parsed as JSON directly.
    # The result (a dict) is stored in args.rename_models.
    parser.add_argument(
        '--rename_models',
        type=json.loads, # Use json.loads as the type directly
        default='{}', # Default is empty JSON string, which json.loads parses to {}
        help='(Passed to minefield_summary.py) JSON string for renaming models (e.g., \'{"Bing": "CoPilot"}\').'
    )

    args = parser.parse_args()

    # Note: args.rename_models now holds the parsed dictionary.

    return args


def get_first_chunk(filepath):
    """
    Reads a file and returns the text content before the first model section,
    by reusing logic from minefield_summary.split_per_model_chunks.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        # This print will go to the current stderr (console or log file)
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return None

    # Reuse minefield_summary's splitting logic.
    # minefield_summary.split_per_model_chunks returns a tuple:
    # (model_chunks[1:], model_names, model_urls, model_chunks[0])
    # We only need the fourth element, which is the first chunk (the prompt).
    try:
        # Pass the entire file content to the summary's splitting function
        # The summary function might raise an error if the input format is unexpected.
        _, _, _, first_chunk = minefield_summary.split_per_model_chunks(content)
        # The returned first_chunk is already stripped by minefield_summary.split_per_model_chunks
        return first_chunk
    except Exception as e:
        # Handle potential errors within the summary's parsing function itself
        # This might happen if the file content is malformed in a way
        # the summary script's regex or splitting logic doesn't expect.
        # This print will go to the current stderr (console or log file)
        print(f"Error parsing file content using minefield_summary logic for {filepath}: {e}", file=sys.stderr)
        # Returning None signals that this file could not be processed
        return None


def gather_prompts(input_folder):
    """
    Scans the input folder for .txt or .md files and extracts the first chunk (prompt) from each.

    Returns:
        dict: A dictionary mapping filename (basename) to the extracted prompt string.
              Returns an empty dictionary if no files are found or parsed successfully.
    """
    prompts = {}
    if not os.path.isdir(input_folder):
        # This print will go to the current stderr (console or log file)
        print(f"Error: Input folder not found or is not a directory: {input_folder}", file=sys.stderr)
        return prompts # Return empty dict on error

    # List files in the directory, filter for actual files ending with .txt or .md
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and (f.endswith('.txt') or f.endswith('.md'))]

    if not files:
        # This print will go to the current stdout (console or log file)
        print(f"Warning: No .txt or .md files found in input folder: {input_folder}")

    for filename in files:
        filepath = os.path.join(input_folder, filename)
        prompt = get_first_chunk(filepath)
        # get_first_chunk returns None on read errors or parsing errors
        if prompt is not None:
            # Use the base filename as the key
            prompts[filename] = prompt
            # This print will go to the current stdout (console or log file)
            print(f"Gathered prompt from '{filename}'")
        else:
             # get_first_chunk already printed an error/warning if it failed
             pass # Skip adding this file if it failed

    print(f"Finished gathering prompts. Found {len(prompts)} valid prompts.")
    return prompts


def create_run_folders(output_base_folder):
    """
    Creates a timestamped run folder within the output_base_folder
    and creates 'model_output' and 'model_analysis' subfolders inside it.

    Args:
        output_base_folder (str): The base directory for runs.

    Returns:
        tuple: A tuple containing (run_folder, model_output_folder, model_analysis_folder),
               or (None, None, None) if folder creation fails.
    """
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    run_folder_name = f"run_{timestamp}"
    run_folder = os.path.join(output_base_folder, run_folder_name)
    model_output_folder = os.path.join(run_folder, 'model_output')
    model_analysis_folder = os.path.join(run_folder, 'model_analysis')

    try:
        # Use exist_ok=True so it doesn't fail if output_base_folder already exists
        os.makedirs(model_output_folder, exist_ok=True)
        os.makedirs(model_analysis_folder, exist_ok=True)
        # Print to the current stdout (console or log file)
        print(f"Created run folder structure: {run_folder}")
        return run_folder, model_output_folder, model_analysis_folder
    except OSError as e:
        # Print error to the current stderr (console or log file)
        print(f"Error creating run folders in {output_base_folder}: {e}", file=sys.stderr)
        return None, None, None


def run_models_on_prompts(prompts, models):
    """
    Runs each specified model on each gathered prompt using the Ollama API.

    Args:
        prompts (dict): Dictionary of filename: prompt string.
        models (list): List of model ID strings.

    Returns:
        dict: A nested dictionary filename: {model_id: response_text}.
              Includes model responses, error messages, or skipped notes.
              Keys correspond to files successfully processed for model runs.
              An empty dictionary indicates no models were run or all attempts failed
              or all prompts were empty.
    """
    results = {}
    total_prompts = len(prompts)
    total_models = len(models)
    total_tasks = total_prompts * total_models # This count includes potential skips if prompt is empty
    current_task = 0

    if not prompts:
        print("No prompts available to run models on.")
        return results

    # If no models are specified, populate results with skipped notes and return early
    if not models:
        print("No models specified to run.")
        for filename in prompts:
             results[filename] = {model_id: "NOTE: No models specified to run." for model_id in models}
        return results

    print(f"Starting model runs: {total_tasks} tasks ({total_prompts} prompts x {total_models} models)")

    # Use a dictionary to store responses for each file
    file_responses = {}

    for filename, prompt in prompts.items():
        file_responses[filename] = {} # Initialize results for this filename
        print(f"\n--- Running models on prompt from '{filename}' ---")
        # Check if prompt is empty after stripping
        if not prompt.strip():
            print(f"Skipping model runs for '{filename}' due to empty prompt.")
            # Store a note that models were skipped for this prompt
            for model_id in models:
                 file_responses[filename][model_id] = "NOTE: Skipped model run due to empty prompt."
            # Increment task count for the skipped runs for accurate total/current display later
            # This ensures the progress counter reaches total_tasks even if prompts are skipped.
            current_task += len(models)
            continue # Move to the next prompt

        # Run models for this non-empty prompt
        for model_id in models:
            current_task += 1
            print(f"[{current_task}/{total_tasks}] Running '{model_id}' on '{filename}'...")
            # Default message in case of failure before storing a specific one
            response_text_on_failure = f"ERROR: Failed to get response from {model_id}. Check console/log for details."

            try:
                # ollama.chat expects a messages list
                response = ollama.chat(model=model_id, messages=[{'role': 'user', 'content': prompt}])
                # Extract the content from the response dictionary
                # Check the structure based on the ollama-python README example: response['message']['content']
                response_text = response.get('message', {}).get('content', str(response))
                if not response_text: # Handle empty content or unexpected response structure
                     # Store an explicit message if model returned empty response
                     response_text = f"Warning: Model '{model_id}' returned empty content for '{filename}'. Full response: {response}"
                print(f"Successfully got response from '{model_id}'.")
                file_responses[filename][model_id] = response_text # Store success or empty warning message

            except ollama.ResponseError as e:
                # Specific error for API issues (e.g., model not found, invalid model)
                print(f"Ollama Response Error running '{model_id}' on '{filename}': Status {e.status_code} - {e.error}", file=sys.stderr)
                file_responses[filename][model_id] = f"ERROR: Ollama Response Error (Status {e.status_code}): {e.error}"
            except Exception as e:
                # Catch any other exceptions (e.g., connection errors, unexpected issues)
                print(f"An unexpected error occurred running '{model_id}' on '{filename}': {e}", file=sys.stderr)
                file_responses[filename][model_id] = f"ERROR: An unexpected error occurred: {e}"
            # Store the error message/note in results so it appears in the output file


    # After iterating through all prompts and models, the file_responses dictionary
    # contains the results/errors/notes for every requested model for every
    # successfully gathered prompt.
    print("\nFinished all model runs.")
    return file_responses


def save_combined_outputs(original_prompts, model_results, output_folder, models_list):
    """
    Formats and saves the original prompt and each model's response into separate
    Markdown files in the specified output folder. Files are named after the
    original input files.

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
    """
    print(f"\n--- Saving combined model outputs to '{output_folder}' ---")

    if not os.path.isdir(output_folder):
         print(f"Error: Output folder not found or is not a directory: {output_folder}", file=sys.stderr)
         return # Cannot save if folder is missing

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
             output_content += "--- Empty Prompt ---\n\n" # Indicate if original prompt was empty


        # Iterate through the *list of requested models* to ensure all are represented
        # in the output file for this prompt. This loop runs for every prompt in original_prompts.
        for model_id in models_list:
            # Use model_id as both name and a placeholder URL for the summary script
            model_name_placeholder = model_id
            model_url_placeholder = model_id # minefield_summary parses this URL field

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
            output_content += f"\n{model_name_placeholder} ({model_url_placeholder}):\n"
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
            # This print will go to the current stderr (console or log file)
            print(f"Error writing output file {output_filepath}: {e}", file=sys.stderr)

    print("\nFinished saving all combined model outputs.")


def run_summary_analysis(model_output_folder, model_analysis_folder, args):
    """
    Executes the minefield_summary.py script as a subprocess to analyze the model outputs.

    Args:
        model_output_folder (str): Path to the folder containing model outputs (input for summary script).
        model_analysis_folder (str): Path to the folder for analysis results (output for summary script).
        args (argparse.Namespace): Parsed command-line arguments from minefield_run.py.
    """
    # This print will go to the current stdout (console or log file)
    print(f"\n--- Running minefield_summary.py for analysis ---")

    # Ensure the analysis output folder exists before starting, even if summary fails.
    os.makedirs(model_analysis_folder, exist_ok=True)

    # Check if there are files to analyze in the model output folder
    # This ensures minefield_summary doesn't run on an empty input folder.
    if not os.path.isdir(model_output_folder) or not os.listdir(model_output_folder):
        print(f"Warning: No model output files found in '{model_output_folder}' for analysis. Skipping summary analysis.")
        return # Exit if no files to process

    # Construct the command to run minefield_summary.py
    # Need to find the path to minefield_summary.py relative to the current script
    # Or assume it's in the same directory or in the PATH.
    # Assuming it's in the same directory for simplicity and robustness.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_script_path = os.path.join(script_dir, 'minefield_summary.py')

    if not os.path.exists(summary_script_path):
        # This print will go to the current stderr (console or log file)
        print(f"Error: minefield_summary.py not found at expected path: {summary_script_path}", file=sys.stderr)
        print("Please ensure minefield_summary.py is in the same directory as minefield_run.py.", file=sys.stderr)
        print(f"Analysis folder '{model_analysis_folder}' will contain only the default files, if any were created before this error.", file=sys.stderr)
        return # Cannot run if script is missing

    command = [
        sys.executable, # Use sys.executable to ensure the same python interpreter is used
        summary_script_path,
        '--input_folder', model_output_folder,
        '--output_folder', model_analysis_folder
    ]

    # Add pass-through arguments if they were set
    if args.skip_descriptor_drop:
        command.append('--skip_descriptor_drop')

    # The --rename_models argument for minefield_summary.py expects a JSON string.
    # args.rename_models holds the parsed dictionary (thanks to type=json.loads).
    # We need to convert this dictionary back into a JSON string to pass it on the command line.
    # Only add the argument if the dictionary is not empty (i.e., not the default {}).
    if args.rename_models: # Check if the dictionary is not empty
       # Dump the dictionary back to a JSON string for the command line argument
       rename_models_json_string = json.dumps(args.rename_models)
       # Use a dedicated argument for the JSON string
       command.extend(['--rename_models', rename_models_json_string])


    print(f"Running command: {' '.join(command)}")

    try:
        # Use capture_output=True and text=True to capture stdout/stderr as strings
        # Then print them ourselves. check=True will raise CalledProcessError on non-zero exit code.
        # We use run() which waits for the subprocess to complete (synchronous).
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("--- minefield_summary.py Standard Output ---") # This output goes to our stdout (log file)
        print(result.stdout)
        if result.stderr:
            # Print stderr to our redirected stderr (log file)
            print("--- minefield_summary.py Standard Error ---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        print("minefield_summary.py finished successfully.") # This output goes to our stdout (log file)

    except FileNotFoundError:
        # This specifically catches if the 'python' executable is not found
        print(f"Error: The '{sys.executable}' command was not found. Make sure Python is installed and in your PATH.", file=sys.stderr)
        print(f"Analysis folder '{model_analysis_folder}' may be empty or incomplete.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        # This catches errors returned by minefield_summary.py itself (non-zero exit code)
        print(f"Error: minefield_summary.py failed with exit code {e.returncode}", file=sys.stderr)
        # Standard output still goes to our stdout (log file)
        print("--- minefield_summary.py Standard Output ---")
        print(e.stdout)
        # Error output goes to our stderr (log file)
        print("--- minefield_summary.py Standard Error ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print(f"Analysis folder '{model_analysis_folder}' may contain incomplete results.", file=sys.stderr)
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        print(f"An unexpected error occurred while running minefield_summary.py: {e}", file=sys.stderr)
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
    # Using original_stdout/stderr explicitly in case print is rebound by an external library
    print("Starting minefield_run process (v3)...", file=original_stdout)

    # 1. Parse arguments
    args = parse_args()
    # Print arguments *before* setting up logging, so they appear initially in the console
    # Using original_stdout explicitly
    print(f"Arguments parsed: {args}", file=original_stdout)


    # 2. Gather prompts from input folder
    # This might print warnings/errors about specific files to the original console
    original_prompts = gather_prompts(args.input_folder)
    if not original_prompts:
        print("No valid prompts found. Exiting.", file=original_stderr)
        return

    # 3. Create run folders
    # This might print messages about folder creation success or error to the original console
    run_folder, model_output_folder, model_analysis_folder = create_run_folders(args.output_base_folder)
    if run_folder is None: # Check if folder creation failed
        print("Failed to create run folders. Exiting.", file=original_stderr)
        return

    # --- START CHANGE 1 & 2 Combined ---
    # Now that we have the run_folder, set up logging and save arguments
    log_filepath = os.path.join(run_folder, 'run.log')
    args_filepath = os.path.join(run_folder, 'run_args.json')

    log_file = None # Initialize log_file handle
    try:
        # Open log file for writing console output
        log_file = open(log_filepath, 'w', encoding='utf-8')
        sys.stdout = log_file # Redirect stdout
        sys.stderr = log_file # Redirect stderr

        # Log confirmation and start time to the new log file
        print(f"Logging console output to '{log_filepath}'...")
        print(f"Run started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Save arguments to JSON file
        args_dict = {
            'input_folder': args.input_folder,
            'output_base_folder': args.output_base_folder,
            'models': args.models, # Already a list
            'skip_descriptor_drop': args.skip_descriptor_drop,
            'rename_models': args.rename_models # Already a dict
        }
        with open(args_filepath, 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, indent=4) # Use indent for readability
        print(f"Saved arguments to '{args_filepath}'") # This print goes to the log file

        # Repeat arguments in log file for completeness and easier debugging
        print(f"Arguments parsed: {args}")

        # --- START v3 Main Logic within logging context ---

        # 4. Run models on gathered prompts
        # These functions and subsequent ones will print to the log file
        model_results = run_models_on_prompts(original_prompts, args.models)

        # 5. Save combined outputs to the model_output folder
        save_combined_outputs(original_prompts, model_results, model_output_folder, args.models)

        # 6. Run minefield_summary.py on the model_output folder
        run_summary_analysis(model_output_folder, model_analysis_folder, args)

        print("\nMinefield run process finished (v3).") # This goes to the log file
        # --- END v3 Main Logic ---

    except Exception as e:
        # Catch any exception that occurs *after* logging is set up
        # Print error and traceback to the redirected stderr (which is the log file)
        print(f"\nFATAL EXCEPTION occurred during run: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Print traceback to log file
        # We could potentially exit here, but letting the 'finally' block run is safer for cleanup

    finally:
        # --- Restore original stdout/stderr and close log file ---
        # Ensure streams are flushed before restoring
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Close the log file if it was successfully opened
        if log_file is not None:
            try:
                log_file.close()
            except Exception as e:
                # Print close errors to the original stderr
                print(f"Error closing log file {log_filepath}: {e}", file=original_stderr)

    # Final message after logging is restored - goes back to original console
    # This confirms the script finished, even if errors occurred before the final print in the log.
    print("Minefield run process completed (check log file in run folder for full output).", file=original_stdout)


if __name__ == "__main__":
    main()