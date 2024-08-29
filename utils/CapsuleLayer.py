"""
Capsule layer (MIND) in pytorch.MINDCapsule
"""

import torch
import torch.nn as nn


class CapsuleLayer(nn.Module):
    """
    Capsule layer (MIND) in pytorch.
    """

    def __init__(self, input_units, output_units, k_max=3, iters=3, init_std=1.0, device="cuda:0"):
        """
        Initialization.
        """
        super(CapsuleLayer, self).__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.k_max = k_max
        self.iters = iters
        self.init_std = init_std
        self.device = device

        # Shared bilinear mapping matrix
        self.bilinear_mapping_matrix = nn.parameter.Parameter(nn.init.normal_(torch.Tensor(input_units, output_units), std=init_std), requires_grad=True,)

    def squash(self, inputs):
        """
        Squash.
        """
        vec_square_norm = torch.sum(torch.square(inputs), dim=-1, keepdim=True)
        scalar_factor = vec_square_norm / (1.0 + vec_square_norm) / torch.sqrt(vec_square_norm + 1e-8)
        vec_squashed = scalar_factor * inputs
        return vec_squashed

    def sequence_mask(self, lengths, max_len=None, dtype=torch.bool):
        """
        Pytorch equivalent for tf.sequence_mask.
        """
        if max_len is None:
            max_len = lengths.max()
        row_vector = torch.arange(0, max_len, 1).to(self.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask.type(dtype)
        return mask

    def forward(self, behavior_embs, seq_len):
        """
        Forward
            behavior_embs: [N, L, D]
            seq_len: [N, 1]
        """
        k_max = self.k_max
        batch_size, max_len, _ = behavior_embs.shape
        seq_len_tile = torch.tile(seq_len, [1, k_max])
        mask = self.sequence_mask(seq_len_tile, max_len)
        pad = torch.ones_like(mask, dtype=torch.float32) * -1e9

        behavior_mappings = torch.matmul(behavior_embs, self.bilinear_mapping_matrix)
        behavior_mappings_detached = behavior_mappings.detach()

        # Behavior-to-Interest (B2I) dynamic routing
        routing_logits = torch.normal(mean=0.0, std=self.init_std, size=(batch_size, k_max, max_len),device=self.device)

        for _ in range(self.iters - 1):
            routing_logits_padded = torch.where(mask, routing_logits, pad)
            weights = torch.softmax(routing_logits_padded, dim=1)
            candidates = torch.matmul(weights, behavior_mappings_detached)
            interests = self.squash(candidates)
            delta_routing_logits = torch.matmul(interests, behavior_mappings_detached.transpose(-2, -1))
            routing_logits += delta_routing_logits

        routing_logits_padded = torch.where(mask, routing_logits, pad)
        weights = torch.softmax(routing_logits_padded, dim=1)
        candidates = torch.matmul(weights, behavior_mappings)
        interests = self.squash(candidates)
        # note: return : B * max_interest* dim
        return interests
