import logging
from model_loader import load_model_and_tokenizer
from graph import create_graph

# --- Configuration ---
MODEL_REPO_ID = "ragunath-ravi/deberta-v3-emotion-classifier"
LOG_FILE = "app.log"

# --- Structured Logging Setup ---
def setup_logging():
    """Configures structured logging to a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=LOG_FILE,
        filemode='w'
    )

def main():
    """Main function to run the CLI application."""
    setup_logging()
    
    print("--- Self-Healing Classification DAG ---")
    logging.info("Application starting.")
    
    try:
        model, tokenizer = load_model_and_tokenizer(MODEL_REPO_ID)
        logging.info(f"Model {MODEL_REPO_ID} loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model. Exiting. Error: {e}")
        print("Fatal: Could not load the model. Please check the repository ID and your connection.")
        return

    app = create_graph(model, tokenizer)
    
    print("\nGraph compiled. Ready for input.")
    print("Type 'exit' or 'quit' to terminate.\n")

    while True:
        try:
            user_input = input("Input: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting application.")
                logging.info("Application terminated by user.")
                break
            
            if not user_input.strip():
                continue

            inputs = {"text_input": user_input}
            logging.info(f"INPUT: '{user_input}'")
            
            result = app.invoke(inputs)
            
            # --- MODIFICATION: Final output formatting ---
            if result.get('fallback_invoked'):
                final_label = result.get('final_label', 'N/A')
                print(f"Final Label: {final_label} (Corrected via user clarification)\n")
            else:
                # If fallback was not invoked, the prediction is the final label
                final_label = result.get('prediction', 'N/A')
                print(f"Final Label: {final_label} (High confidence)\n")

            log_message = (
                f"FINAL_DECISION: text='{result['text_input']}' | "
                f"prediction='{result['prediction']}' | "
                f"confidence={result.get('confidence', 0):.4f} | "
                f"fallback_invoked={result.get('fallback_invoked', False)} | "
                f"final_label='{final_label}'"
            )
            logging.info(log_message)

        except KeyboardInterrupt:
            print("\nExiting application.")
            logging.info("Application interrupted by user.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
            print("An unexpected error occurred. Check app.log for details.")

if __name__ == "__main__":
    main()