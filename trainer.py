# Install required packages (run this cell first)
!pip install transformers torch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from datetime import datetime
from transformers import BertTokenizer
from google.colab import drive
import torch.cuda.amp as amp

class ChatbotConfig:
    def __init__(self):
        self.hidden_size = 512
        self.num_layers = 6
        self.num_heads = 8
        self.dropout = 0.1
        self.max_length = 128
        self.vocab_size = 30522  # BERT tokenizer vocabulary size
        self.batch_size = 64     # Increased batch size for GPU
        self.learning_rate = 3e-4
        self.checkpoint_dir = "/content/drive/MyDrive/chatbot/checkpoints"
        self.history_file = "/content/drive/MyDrive/chatbot/conversation_history.json"

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

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        encoded = self.tokenizer.encode_plus(
            conversation["input"],
            conversation["response"],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
        }

class Chatbot:
    def __init__(self, config=None):
        self.config = config or ChatbotConfig()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TransformerEncoder(self.config)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.scaler = amp.GradScaler()

        print(f"Using device: {self.device}")

        # Mount Google Drive and create directories
        drive.mount('/content/drive')
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Load conversation history if it exists
        self.conversation_history = self.load_history()

    def load_history(self):
        try:
            with open(self.config.history_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_history(self):
        os.makedirs(os.path.dirname(self.config.history_file), exist_ok=True)
        with open(self.config.history_file, 'w') as f:
            json.dump(self.conversation_history, f)

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def train(self, train_conversations, num_epochs=10):
        print("Preparing dataset...")
        dataset = ConversationDataset(train_conversations, self.tokenizer, self.config.max_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                self.optimizer.zero_grad()

                # Mixed precision training
                with amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    targets = input_ids[:, 1:].contiguous()
                    outputs = outputs[:, :-1].contiguous()
                    loss = criterion(outputs.view(-1, self.config.vocab_size), targets.view(-1))

                # Scale loss and backpropagate
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

            # Save checkpoint to Google Drive
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, avg_loss)

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
            for _ in range(max_length):
                try:
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

                except RuntimeError as e:
                    print(f"Error during response generation: {str(e)}")
                    return "I apologize, but I encountered an error generating a response."

        response = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response
        })
        self.save_history()

        return response

def prepare_training_data():
    """
    Example function to prepare training data.
    In practice, you would load your own conversation dataset.
    """
    return [
        {
            "input": "Hello, how are you?",
            "response": "I'm doing well, thank you for asking! How can I help you today?"
        },
        {
            "input": "What's the weather like?",
            "response": "I don't have access to real-time weather information. You might want to check a weather service or app for accurate information."
        },
        {
            "input": "Tell me a joke",
            "response": "Why did the scarecrow win an award? Because he was outstanding in his field!"
        },
        {
            "input": "What's your favorite color?",
            "response": "I don't have personal preferences, but I can discuss different colors and their meanings if you'd like!"
        },
        {
            "input": "How do I learn programming?",
            "response": "Learning programming is a journey that starts with choosing a language, practicing regularly, and working on projects. Python is often recommended for beginners."
        },
        {
            "input": "Say hello"
            "response": "hello"
        }
    
        
        
        
        
        
        
        
        
        ]

def main():
    # Initialize chatbot
    print("Initializing chatbot...")
    config = ChatbotConfig()
    chatbot = Chatbot(config)

    # Prepare training data
    print("Preparing training data...")
    train_data = prepare_training_data()

    # Train the model
    print("Starting training...")
    chatbot.train(train_data, num_epochs=30)

    # Save the final model
    chatbot.save_checkpoint("final", 0.0)
    print("Training complete! Model saved to Google Drive")

if __name__ == "__main__":
    main()
