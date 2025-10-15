import torch
import torch.nn.functional as F
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Constants ---
CONFIDENCE_THRESHOLD = 0.80

# --- State Definition ---
class GraphState(TypedDict):
    """
    Defines the state that flows through the graph.
    """
    text_input: str
    prediction: Optional[str]
    confidence: Optional[float]
    fallback_invoked: bool
    final_label: Optional[str]

# --- Node Functions ---
def inference_node(state: GraphState, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> GraphState:
    """
    Performs inference and prints the result in the specified format.
    """
    device = model.device
    text = state['text_input']
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True,max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    top_idx = int(probs.argmax())
    predicted_label = model.config.id2label[top_idx]
    confidence = float(probs[top_idx])
    
    print(f"[InferenceNode] Predicted label: {predicted_label} | Confidence: {confidence:.0%}")
    
    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "fallback_invoked": False,
    }

def fallback_node(state: GraphState, model: AutoModelForSequenceClassification) -> GraphState:
    """
    Handles low-confidence predictions by asking the user for the correct label.
    """
    valid_labels = list(model.config.id2label.values())
    final_label = ""

    # MODIFICATION: This prompt is now a single, direct question.
    prompt = f"[FallbackNode] Could you clarify your intent? The model predicted '{state['prediction']}': "

    while True:
        user_input = input(prompt).strip().lower()
        
        # MODIFICATION: Echo user input to match the assignment's 'User:' line format.
        # This line will only be printed after the user provides input.
        print(f"User: {user_input}")

        if user_input in valid_labels:
            final_label = user_input
            break
        elif not user_input: # Allow user to accept original prediction
            final_label = state['prediction']
            break
        else:
            # Provide guidance only on error
            print(f"  Invalid label. Please choose from: {', '.join(valid_labels)}")
            prompt = "  Enter the correct label: " # Use a simpler prompt for retries

    return {
        "final_label": final_label,
        "fallback_invoked": True,
    }

# --- Conditional Edge Logic ---
def confidence_check_edge(state: GraphState) -> str:
    """
    Routes the workflow based on confidence and prints the status.
    """
    if state['confidence'] < CONFIDENCE_THRESHOLD:
        print(f"[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
        return "fallback"
    else:
        state['final_label'] = state['prediction']
        return END

# --- Graph Construction ---
def create_graph(model, tokenizer):
    """
    Creates and compiles the LangGraph workflow.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("inference", lambda state: inference_node(state, model, tokenizer))
    workflow.add_node("fallback", lambda state: fallback_node(state, model))

    workflow.set_entry_point("inference")

    workflow.add_conditional_edges(
        "inference",
        confidence_check_edge,
    )

    workflow.add_edge("fallback", END)
    
    return workflow.compile()