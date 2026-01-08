import torch
import torch.nn as nn

class TemporalAutoencoder(nn.Module):
    """
    GRU-based Autoencoder for capturing temporal clinical patterns.
    This model encodes a sequence of visits into a fixed-size latent vector.
    """
    def __init__(self, input_dim=50, hidden_dim=256, latent_dim=128):
        super(TemporalAutoencoder, self).__init__()
        
        # Encoder: GRU to process sequence and extract hidden state
        self.gru_enc = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Mapping latent back to sequence space for reconstruction
        self.fc_upscale = nn.Linear(latent_dim, hidden_dim)
        # Using another GRU or Linear layers to reconstruct the input sequence
        self.gru_dec = nn.GRU(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        """
        Forward pass for temporal encoding.
        Input x shape: (Batch, Time, Features)
        """
        # Encoding phase
        # _, h_n = self.gru_enc(x) -> h_n shape: (1, Batch, Hidden)
        _, h_n = self.gru_enc(x)
        
        # Get latent representation from the final hidden state
        latent = self.fc_latent(h_n.squeeze(0))
        
        # Decoding phase (Reconstruction for unsupervised learning)
        # We repeat the hidden state to match the sequence length logic if needed,
        # but here we use a simplified version for reconstruction loss.
        h_d = self.fc_upscale(latent).unsqueeze(0)
        
        # We pass x as a placeholder or use zeros to reconstruct
        reconstructed, _ = self.gru_dec(x, h_d)
        
        return latent, reconstructed

class CategoricalEmbedder(nn.Module):
    """
    Embedding layer for categorical clinical codes (ICD, Medications).
    Inspired by Med2Vec/Deepr architectures.
    """
    def __init__(self, vocab_size=1000, embed_dim=128):
        super(CategoricalEmbedder, self).__init__()
        # Linear layer acting as an embedding matrix for multi-hot encoded inputs
        self.embedding = nn.Linear(vocab_size, embed_dim)
        
    def forward(self, x):
        """
        Input x: Multi-hot encoded vector (Batch, Vocab_Size)
        Output: Dense embedding vector
        """
        return torch.relu(self.embedding(x))
