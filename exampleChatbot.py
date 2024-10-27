import torch
import torch.nn as nn
from transformers import BertTokenizer
from datetime import datetime
import json
import os

class ChatbotConfig:
    def __init__(self):
        self.hidden_size = 512
        self.num_layers = 6
        self.num_heads = 8
        self.dropout = 0.1
        self.max_length = 128
        self.vocab_size = 30522  # BERT tokenizer vocabulary size
        self.learning_rate = 3e-4
        self.history_file = "conversation_history.json"

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.fc_out = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        if mask is not None:
            padding_mask = ~mask.bool()
        else:
            padding_mask = None
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.fc_out(x)

class Chatbot:
    def __init__(self, checkpoint_path, config=None):
        self.config = config or ChatbotConfig()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TransformerEncoder(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
        print("Loading checkpoint...")
        self.load_checkpoint(checkpoint_path)
        
        # Load conversation history
        self.conversation_history = self.load_history()
        
    def load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Load only the model state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Successfully loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            raise
    
    def load_history(self):
        try:
            with open(self.config.history_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
            
    def save_history(self):
        with open(self.config.history_file, 'w') as f:
            json.dump(self.conversation_history, f)

    def generate_response(self, user_input, max_length=50):
        self.model.eval()
        
        encoded = self.tokenizer.encode_plus(
            user_input,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            try:
                for _ in range(max_length):
                    outputs = self.model(input_ids, attention_mask)
                    next_token_logits = outputs[:, -1, :]
                    
                    temperature = 0.7
                    scaled_logits = next_token_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=self.device)
                    ], dim=-1)
                    
                    if next_token.item() == self.tokenizer.sep_token_id:
                        break
                        
                response = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                
            except Exception as e:
                print(f"Error during response generation: {str(e)}")
                response = "I apologize, but I encountered an error generating a response."
        
        # Save conversation to history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response
        })
        self.save_history()
        
        return response

def chat_interface(checkpoint_path):
    """Interactive chat interface for the trained chatbot."""
    try:
        chatbot = Chatbot(checkpoint_path)
        print("\nChatbot loaded and ready! Type 'quit' to exit.")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break
                
            response = chatbot.generate_response(user_input)
            print(f"Bot: {response}")
            
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")

if __name__ == "__main__":
    checkpoint_path = "checkpoint_epoch_final.pt"  # Path to your checkpoint file
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found!")
    else:
        chat_interface(checkpoint_path)
