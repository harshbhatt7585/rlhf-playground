from transformers import AutoTokenizer, AutoModelForCausalLM

def load_base_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model



if __name__ == "__main__":
    # Example usage
    tokenizer, model = load_base_model()
    print(f"Loaded model: {model.config._name_or_path}")
    print(f"Tokenizer: {tokenizer.name_or_path}")
    # You can now use the tokenizer and model for inference or training