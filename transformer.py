#!/usr/bin/env python
# coding: utf-8

# Implementation of Transformer model.
# 
# This module contains classes and functions which implement the main parts of
# the Transformer model, as presented in article Attention Is All You Need
# by Vaswani et al.

import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_vocabulary: int, d_model: int) -> None:
        super().__init__()
        self.d_vocabulary = d_vocabulary
        self.d_model = d_model
        self.embedding = nn.Embedding(self.d_vocabulary, self.d_model)
    
    def forward(self, input: torch.Tensor):
        return self.embedding(input) * math.sqrt(self.d_model)


class Linear(nn.Module):
    def __init__(self, d_model: int, d_vocabulary: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_vocabulary)
    
    def forward(self, input: torch.Tensor):
        return self.linear(input)


class Softmax(nn.Module):
    def __init__(self, dim: int = None) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
    
    def forward(self, input: torch.Tensor):
        return self.softmax(input)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, d_sequence: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_sequence = d_sequence
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zero(d_sequence, d_model)
        pos = torch.arange(0, d_sequence, dtype=torch.float).unsqueeze(1)
        # Use exp and log to increase performance.
        denominator = torch.exp(torch.arange(0, d_model, 2).float() / d_model * math.log(10000))
        pe[:, 0::2] = torch.sin(pos / denominator)
        pe[:, 1::2] = torch.cos(pos / denominator)
        
        # Add batch dimension for parallel processing of sequences.
        pe = pe.unsqueeze(0)
        
        # Store positional encoding parameters for future analysis.
        self.register_buffer("pe", pe)
    
    def forward(self, input: torch.Tensor):
        # No need to learn positional encoding parameters.
        input = input + (self.pe[:, :input.shape[1], :]).requires_grad_(False)
        return self.dropout(input)


class Norm(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, input: torch.Tensor):
        # Dimension is kept to allow broadcasting.
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, correction=0, keepdim=True)
        return self.gain / (std + self.eps) * (input - mean) + self.bias


class ResidualConnection(nn.Module):
    def __init__(self, eps: float, dropout: float) -> None:
        super().__init__()
        self.norm = Norm(eps)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor, sublayer: nn.Module):
        return self.norm(input + self.dropout(sublayer(input)))


class Feedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, input: torch.Tensor):
        return self.linear_2(self.relu(self.linear_1(input)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        # Dimension of embedding is supposed to be divisible by number of heads.
        self.d_k = d_model // h
        self.d_v = self.d_k
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.softmax = Softmax(dim=3)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.BoolTensor):
        query = self.W_Q(Q)
        key = self.W_K(K)
        value = self.W_V(V)
        
        # Prepare input vectors for attention heads.
        query = query.reshape(query.shape[0], query.shape[1], self.h, self.d_k)
        key = key.reshape(key.shape[0], key.shape[1], self.h, self.d_k)
        value = value.reshape(value.shape[0], value.shape[1], self.h, self.d_v)
        
        attention = torch.matmul(query.transpose(1, 2), key.transpose(1, 2).transpose(2, 3))
        # Mask to saturate to zero the softmax function.
        if mask is not None:
            attention.masked_fill_(mask == 0, -1e15)
        attention = self.softmax(attention / math.sqrt(self.d_k))
        attention = torch.matmul(attention, value)
        
        # Concatenate heads.
        output = attention.transpose(1, 2)
        output.reshape(output.shape[0], output.shape[1], self.h * self.d_k)
        return self.W_O(output)

