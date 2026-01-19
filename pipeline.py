import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_filename, output_dir="pipeline_logs"):
    """
    Executes a Jupyter notebook and saves the executed version with outputs.
    """
    print(f"Starting execution of: {notebook_filename}...")
    
    # Ensure output directory exists for logs
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the notebook
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
        
    # Configure the executor
    ep = ExecutePreprocessor(timeout=600, kernel_name='venv')
    
    try:
        # Run the notebook
        ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
        
        # Save the executed notebook (useful for debugging)
        output_filename = os.path.join(output_dir, f"executed_{notebook_filename}")
        with open(output_filename, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        print(f"Successfully ran {notebook_filename}. Log saved to {output_filename}")
        return True
        
    except Exception as e:
        print(f"Error executing {notebook_filename}")
        print(f"   Reason: {e}")
        return False

def main():
    pipeline_steps = [
        "/Users/nikitha/Documents/flight-delay-prediction-ice/notebooks/data_cleaning_feature_engineering.ipynb",
        "/Users/nikitha/Documents/flight-delay-prediction-ice/notebooks/flight_delay_eda.ipynb"
        "/Users/nikitha/Documents/flight-delay-prediction-ice/notebooks/flight_delay_ml_modeling.ipynb"
    ]
    
    print("Starting Flight Delay ML Pipeline")

    for step in pipeline_steps:
        if not os.path.exists(step):
            print(f"File not found: {step}")
            return

        success = run_notebook(step)
        if not success:
            print(" Pipeline stopped due to error.")
            return

    print("Pipeline completed successfully!")
    print("* Data cleaned and features engineered.")
    print("* Models trained and artifacts saved.")
    print("* You are now ready to run 'streamlit run app.py'")

if __name__ == "__main__":
    main()