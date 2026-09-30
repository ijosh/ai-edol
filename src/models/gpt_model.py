"""
GPT model architecture for smart meter data generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import logging
import numpy as np
from pathlib import Path

class EmbeddingModel(nn.Module):
    """
    This is an isolated version of the "frontend" of the model. IE just the code to go from input -> tokens -> embeddings
    """
    def __init__(self, 
        vocab_sizes: List[int],
        d_model: int,
        context_length: int
    ):
        super().__init__()
        
        self.num_input_features = len(vocab_sizes)
        self.d_model = d_model
        self.context_length = context_length
        
        # Embedding layers for each input feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for vocab_size in vocab_sizes
        ])
        
        # Learnable positional embedding
        self.pos_embedding = nn.Embedding(context_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed each feature separately and sum them
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            
            # Dont attempt to include runtime validation in the onnx export
            if not torch.onnx.is_in_onnx_export():
                max_index = x[:, :, i].max().item()
                vocab_size = emb_layer.num_embeddings
                if max_index >= vocab_size:
                    logging.debug(f'Feature {i} has index out of range. Max: {max_index}, vocab: {vocab_size}')
                    raise ValueError(f'Input feature {i} has index ({max_index}) exceeding vocab size ({vocab_size})')
            
            embeddings.append(emb_layer(x[:, :, i].long()))
        
        x = sum(embeddings)  # [batch_size, context_length, d_model]
        
        # Add learnable positional embedding
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_encoding = self.pos_embedding(positions)
        x = x + pos_encoding

        return x # [B, T, d_model]

class EmbeddableComputeModel(nn.Module):
    """
    This is an isolated version of the computationally heavy part of the model.
    It is separated so that it can be exported to onnx and deployed on the hailo10h efficiently. 
    """
    def __init__(self, trained_model, seq_len=336):
        super().__init__()

        self.seq_len = trained_model.context_length
        self.d_model = trained_model.d_model

        self.transformer_layers = (
            trained_model.transformer_decoder_layers
        )

        self.output_layer = trained_model.output_layer

        causal_mask = torch.zeros(
            self.seq_len,
            self.seq_len,
            dtype=torch.float32
        )

        causal_mask = causal_mask.masked_fill(
            torch.triu(
                torch.ones(
                    self.seq_len,
                    self.seq_len,
                    dtype=torch.bool
                ),
                diagonal=1
            ),
            float("-inf")
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False
        )


    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x, tgt_mask=self.causal_mask)
        
        # Only take the final timestep
        #x = x[:, self.seq_len - 1, :]

        # [B,32] -> [B,66]
        x = self.output_layer(x)

        return x #[B, 66]


class CustomTransformerDecoderLayer(nn.Module):
    """Custom Transformer Decoder Layer with prenorm architecture."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        """
        Initialize CustomTransformerDecoderLayer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
        """
        super(CustomTransformerDecoderLayer, self).__init__()
        
        print(f"Model Dimension {d_model}\nnhead {nhead}\ndim_feedforward {dim_feedforward}\ndropout {dropout}")
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, tgt: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the layer.
        
        Args:
            tgt: Target tensor
            tgt_mask: Attention mask
            
        Returns:
            Transformed tensor
        """
        # Prenorm: apply layer normalisation before self-attention
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # Prenorm: apply layer normalisation before feedforward network
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        
        return tgt


class GPTSmartMeterModel(nn.Module):
    """
    GPT model for smart meter data generation with single-output mode.

    This model can be configured to generate either electricity OR gas,
    enabling separate specialized models for each target variable while
    maintaining cross-fuel context through input features.

    Architecture Modes:
    - Electricity Model: Generates electricity conditioned on gas history + context
    - Gas Model: Generates gas conditioned on electricity history + context

    Feature Ordering (11 features):
    - Index 0: Electricity (INPUT AND OUTPUT for elec model, INPUT for gas model)
    - Index 1: Gas (INPUT for elec model, INPUT AND OUTPUT for gas model)
    - Index 2: Temperature (INPUT for both)
    - Index 3: Solar radiation (INPUT for both)
    - Index 4: Half-hour of day (INPUT for both, 1-48)
    - Index 5: Day of week (INPUT for both, 0-6)
    - Index 6: Month (INPUT for both, 1-12)
    - Index 7: building_type (INPUT CONDITIONING for both)
    - Index 8: age_built (INPUT CONDITIONING for both)
    - Index 9: num_rooms (INPUT CONDITIONING for both)
    - Index 10: num_occs (INPUT CONDITIONING for both)

    Simplified Conditioning Approach:
    - Reduced from 14 to 4 conditioning variables for improved fidelity
    - Selected variables: building_type, age_built, num_rooms, num_occs
    - No masking - all conditioning variables always present

    Usage:
    - Two separate models trained independently (electricity and gas)
    - Both models see cross-fuel context during training (teacher forcing)
    - During inference, DualModelGenerator coordinates parallel generation
    - Preserves elec-gas correlations through historical context
    """
    
    def __init__(
        self,
        vocab_sizes: List[int],
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        context_length: int,
        dropout: float = 0.1,
        output_variable: str = 'electricity'
    ):
        """
        Initialize GPTSmartMeterModel.

        Args:
            vocab_sizes: List of vocabulary sizes for each input feature (11 features).
                        Computed from tokenizer metadata, not configured manually.
            d_model: Model dimension (embedding and hidden layer size)
            nhead: Number of attention heads
            num_layers: Number of transformer decoder layers
            dim_feedforward: Feedforward network dimension
            context_length: Maximum sequence length (e.g., 2048)
            dropout: Dropout probability for regularization
            output_variable: Target variable to generate - either 'electricity' or 'gas'.
                           Determines which output head is created.

        Raises:
            ValueError: If output_variable is not 'electricity' or 'gas'
            AssertionError: If vocab_sizes does not have exactly 11 elements
        """
        super(GPTSmartMeterModel, self).__init__()

        # Validate vocab_sizes length (simplified conditioning: 4 variables instead of 14)
        assert len(vocab_sizes) == 11, f"Expected 11 vocab_sizes, got {len(vocab_sizes)}"
        
        self.output_variable = output_variable

        self.num_input_features = len(vocab_sizes)
        self.d_model = d_model
        self.context_length = context_length

        self.embedding_model = EmbeddingModel(
            vocab_sizes=vocab_sizes,
            d_model=d_model,
            context_length=context_length,
        )
        
        # Register lower triangular mask as buffer for efficiency
        self.register_buffer(
            'tril_mask', 
            torch.tril(torch.ones(context_length, context_length))
        )
        
        # Custom Transformer Decoder layers
        print(f"NUM LAYERS IS {num_layers}")
        self.transformer_decoder_layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) 
            for _ in range(num_layers)
        ])
        
        if output_variable == 'electricity':
            self.output_layer = nn.Linear(d_model, vocab_sizes[0])
        elif output_variable == 'gas':
            self.output_layer = nn.Linear(d_model, vocab_sizes[1])
        else:
            raise ValueError(f"Unsupported output variable: {output_variable}")


    def process_compute_hailo(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Running via Hailo backend is not yet implemented")
        return 
    
    def process_compute_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        # Generate causal mask
        subsequent_mask = self.tril_mask[:seq_len, :seq_len]
        subsequent_mask = subsequent_mask.masked_fill(subsequent_mask == 0, float('-inf'))
        
        # Pass through transformer layers
        for layer in self.transformer_decoder_layers:
            x = layer(x, tgt_mask=subsequent_mask)
            
        # Project to output vocabularies (electricity or gas depending on config)
        output = self.output_layer(x)          
        return output


    def process_compute(self, x: torch.Tensor, backend: str = "cpu") -> torch.Tensor:

        # Always train on cpu. Optionally run inference on the hailo card
        if backend == "cpu" or self.training:
            return self.process_compute_pytorch(x)        
        elif backend == "hailo":
            return self.process_compute_hailo(x)
        else:
            raise ValueError(f"Unsupported Backend: {backend}")
        
    def forward(self, x: torch.Tensor, backend = "cpu", log: bool = True) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, num_input_features]
               where num_input_features = 11 (elec, gas, temp, solar, hh, dow, month, + 4 conditioning)

        Returns:
            Output logits of shape [batch_size, seq_len, vocab_size]
            where vocab_size is either vocab_sizes[0] (electricity) or vocab_sizes[1] (gas)
            depending on output_variable configuration.

        Note:
            This method only performs forward pass. Autoregressive generation is handled
            by DualModelGenerator which coordinates both electricity and gas models.
        """

        embeddings = self.embedding_model(x)
        output = self.process_compute(embeddings, backend=backend)

        if log:
            assert backend == "cpu", "Logging with backend != cpu"

            # Log inputs as integer (token ids)
            x.detach().cpu().numpy().astype("<i4").tofile("input.bin")

            # Log embeddings and outputs as f32
            embeddings.detach().cpu().numpy().astype("<f4").tofile("embeddings.bin")
            output.detach().cpu().numpy().astype("<f4").tofile("output.bin")


        return output

    def print_model_info(self):
        print(self)
    
    
    def create_embeddable_model(self, output_dir):
        """
        Run this once the model is trained to extract all the transformer layers and put them in an onnx file
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.eval(); # Put model in evaluation mode to disable dropout etc
        
        # Export the embedding model
        dummy_input = torch.zeros(
            1,
            self.context_length,
            self.num_input_features,
            dtype=torch.int32
        )

        # Export model
        torch.onnx.export(
            self.embedding_model,
            (dummy_input,),
            output_dir / "embedding_model.onnx",
            input_names=["tokens"],
            output_names=["embeddings"],
            opset_version=17,
            do_constant_folding=True
        )
        
        print(f"Exported ONNX embedding model")

        # also export the model weights as binary so that we can ingest them in C and create our own implementation of the frontend of the model. 
        for i, embedding in enumerate(self.embedding_model.embeddings):
            embedding.weight.detach().cpu().numpy().astype("<f4").tofile(output_dir / f"embedding_{i}.bin")

        self.embedding_model.pos_embedding.weight.detach().cpu().numpy().astype("<f4").tofile(output_dir / "pos_embedding.bin")

        print(f"Exported embedding model raw weights")

        # Next export the compute model
        compute_model = EmbeddableComputeModel(self)
        compute_model.eval()
        compute_model_input = self.embedding_model(dummy_input)

        # Test it matches pytorch compute model
        with torch.no_grad():
            output_ebd = compute_model(compute_model_input)
            output_cpu = self.process_compute_pytorch(compute_model_input)

            # Compare cpu model with last output of cpu model
            torch.testing.assert_close(
                output_ebd,
                output_cpu
            ) 

        # Export model
        torch.onnx.export(
            compute_model,
            (compute_model_input,),
            output_dir / "compute_model.onnx",
            input_names=["embedded_input"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True
        )

        print(f"Exported ONNX compute model")

        return

