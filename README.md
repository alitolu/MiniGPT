# MiniGPT - C# LLM Framework

## Overview
MiniGPT is a ground-up implementation of a Transformer-based Large Language Model (LLM) engine, written entirely in C#. Its goal is to demonstrate the internal mechanics of modern LLMs, including training pipelines, inference optimization, and agentic capabilities, without relying on external machine learning libraries.

## Architecture

### 1. Core Layer
- **Tensor:** A custom tensor implementation supporting FP32, FP16, INT8, and Q4 quantization. Includes automatic differentiation structures.
- **Ops:** Fundamental linear algebra operations (MatMul, Add, ReLU, Softmax) optimized for CPU execution.
- **Autograd:** Gradient computation graph support.
- **FlashAttention:** A CPU-optimized implementation of the FlashAttention algorithm (O(N) memory complexity) using online softmax and row-wise streaming.

### 2. Neural Network (NN) Layers
- **Linear:** Fully connected layers with quantization support (dynamic weight compression).
- **LayerNorm:** Standardization layer for training stability (Pre-Norm architecture).
- **PositionalEncoding:** Sinusoidal encoding for sequence order awareness.
- **MultiHeadAttention:** Parallel attention mechanism supporting both classical and FlashAttention backends.
- **TransformerBlock:** The core building block combining attention, feed-forward networks (MLP), and residual connections.

### 3. Optimization
- **AdamW:** A stochastic gradient descent method with weight decay, momentum, and bias correction.
- **CrossEntropy:** Loss function computation for next-token prediction.

### 4. Tokenizer
- **BPETokenizer:** Byte Pair Encoding tokenizer trained from scratch on raw text corpus.
- **TokenizerBinLoader:** Support for loading LLaMA-compatible tokenizer binary formats.

### 5. Engine
- **MiniGPTModel:** The main model architecture orchestrating embeddings, transformer blocks, and output heads. Supports Train and Inference modes.
- **Trainer:** Manages the training loop including forward pass, backward pass (backpropagation), and optimizer step.
- **ChatEngine:** Handles text generation with temperature sampling, Top-K/Top-P logic, and context management.
- **KVCache:** Key-Value caching system with PagedAttention and Sliding Window mechanism for efficient long-context inference.

### 6. Data Pipeline
- **StreamingDataset:** Efficiently streams large text files line-by-line to minimize RAM usage during training.
- **DataLoader:** Batches and shuffles data for stochastic training.

### 7. Agent Layer
- **AgentLoop:** Orchestrates the interaction between the LLM, external tools, and memory.
- **ToolRegistry:** Plugin system for registering and executing external tools (e.g., Calculator).
- **RAG Pipeline:** Retrieval-Augmented Generation using a simple Vector Store and Embedding Model.
- **Memory:** Conversation history management.

### 8. Deployment & Interface
- **ApiServer:** ASP.NET Core minimal API providing REST and SSE (Server-Sent Events) endpoints.
- **WebUI:** A dark-themed, responsive chat interface communicating with the API.
- **GGUFExporter:** Export functionality to save model weights in GGUF-compatible binary format.

## Features

- **Training:** End-to-end training pipeline from raw text.
- **Quantization:** Support for 8-bit and 4-bit weight quantization to reduce memory footprint.
- **Optimization:** PagedAttention and FlashAttention for faster inference and lower memory usage.
- **Streaming:** Token-by-token generation streaming via API.
- **Agent Capabilities:** Tool use (Function Calling) and RAG support.
- **Cross-Platform:** Runs on any platform supporting .NET 8.0.

## Usage

Run the application using the following command:

    dotnet run

### Modes

1. **Train:**
   Trains a new model from scratch using `dataset.txt`.
   - The system will process the dataset, train the tokenizer, and run the training loop.
   - Checkpoints are saved periodically (e.g., `model_epoch_0.ckpt`).
   - Final quantized model is saved as `model_final.ckpt`.

2. **Chat (Console):**
   Interactive console chat with the trained model.
   - Loads `model_final.ckpt`.
   - Applies quantization for inference.

3. **Serve (Web UI + API):**
   Starts a local web server at `http://localhost:5000`.
   - Access the Web UI via browser.
   - API Endpoint: `POST /chat`
   - Stream Endpoint: `GET /stream`

4. **Agent:**
   Runs the model in Agent mode with tool access.
   - Capable of using a Calculator tool for math operations.
   - Uses RAG to retrieve project-specific knowledge.
   - Maintains conversation memory.

## Requirements
- .NET 8.0 SDK

## Directory Structure

- **Core/**: Low-level tensor operations.
- **NN/**: Neural network layers.
- **Optim/**: Optimization algorithms.
- **Tokenizer/**: Text processing.
- **Model/**: Model architecture definition.
- **Engine/**: Training and inference logic.
- **Data/**: Dataset handling.
- **Agent/**: high-level agent logic.
- **Server/**: API and Web server.
- **Tools/**: External tool implementations.
- **Embeddings/**: RAG components.
- **Memory/**: Context management.
