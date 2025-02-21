# Custom Models Using Custom Layers

This project demonstrates the implementation and training of two distinct deep learning models built entirely using custom layer functions. The models are constructed without leveraging high-level neural network libraries for core operations. Instead, every operation—ranging from convolutions to activations—is built from basic tensor operations. This approach encourages a deep understanding of the underlying mathematical processes that drive modern deep learning techniques.

## Project Overview

In this project, two models have been implemented:

1. **Custom Convolutional Neural Network (CNN) for MNIST**  
   This model is designed to classify handwritten digits from the MNIST dataset. It uses custom implementations for all core operations, including 2D convolution, ReLU activation, max pooling, flattening, and a fully connected layer. The model’s architecture has been purposefully kept simple, focusing on the essential components of a convolutional network.

2. **Custom Transformer (Decoder-Only) for Text Data**  
   This model implements a simplified version of a Transformer model, particularly a decoder-only variant. It incorporates a self-attention mechanism and a feed-forward network. All the computations, such as the linear transformations for query, key, and value generation, the scaling and softmax in the attention mechanism, and the final projection to the vocabulary size, are done using custom functions. This model illustrates the mechanics behind attention-based architectures.

---

## What Is Being Done and How

In this project, every computational step is manually implemented. 

### Custom CNN Model for MNIST

The CNN model is built with the following components:

1. **Custom Convolution Operation:**  
   Instead of using `torch.nn.Conv2d`, the convolution is implemented using explicit loops to slide over the input tensor and compute the dot product between the kernel and the corresponding patch from the input.
   ```python
   def conv2d_custom(x, weight, bias, stride=1, padding=0):
       # Apply padding manually
       x_padded = pad2d(x, padding)
       N, C_in, H, W = x_padded.shape
       C_out, _, kH, kW = weight.shape
       H_out = (H - kH) // stride + 1
       W_out = (W - kW) // stride + 1
       out = torch.zeros((N, C_out, H_out, W_out), device=x.device)
       for n in range(N):
           for c in range(C_out):
               for i in range(H_out):
                   for j in range(W_out):
                       h_start = i * stride
                       w_start = j * stride
                       patch = x_padded[n, :, h_start:h_start+kH, w_start:w_start+kW]
                       out[n, c, i, j] = torch.sum(patch * weight[c]) + bias[c]
       return out
   ```
   This function manually applies convolution over the input and shows clearly how each kernel is applied, emphasizing the fundamental mathematical operations behind a convolution layer.

2. **Custom Activation and Pooling:**  
   - **ReLU Activation:** Implemented using tensor clamping.
     ```python
     def relu(x):
         return torch.clamp(x, min=0)
     ```
   - **Max Pooling:** The max pooling function iterates over each window and extracts the maximum value.
     ```python
     def max_pool2d_custom(x, kernel_size=2, stride=2):
         N, C, H, W = x.shape
         H_out = (H - kernel_size) // stride + 1
         W_out = (W - kernel_size) // stride + 1
         out = torch.zeros((N, C, H_out, W_out), device=x.device)
         for n in range(N):
             for c in range(C):
                 for i in range(H_out):
                     for j in range(W_out):
                         h_start = i * stride
                         w_start = j * stride
                         patch = x[n, c, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                         out[n, c, i, j] = torch.max(patch)
         return out
     ```

3. **Fully Connected Layer:**  
   The fully connected layer is also built from scratch without any reliance on `torch.nn.Linear`.
   ```python
   def linear_custom(x, weight, bias):
       return x @ weight + bias
   ```
   This matrix multiplication encapsulates the core idea of a fully connected layer.

4. **Model Architecture and Training:**  
   The CNN is structured with two convolutional blocks followed by a flattening operation and a fully connected layer. The training loop manually computes the forward pass, loss (using a custom cross-entropy calculation), and backpropagation using gradient descent.
   ```python
   class CustomCNN:
       def __init__(self):
           # Define parameters for two convolutional layers and one fully connected layer
           self.conv1_weight = torch.nn.Parameter(torch.randn(6, 1, 5, 5) * 0.1)
           self.conv1_bias = torch.nn.Parameter(torch.zeros(6))
           self.conv2_weight = torch.nn.Parameter(torch.randn(12, 6, 5, 5) * 0.1)
           self.conv2_bias = torch.nn.Parameter(torch.zeros(12))
           fc_in_features = 12 * 4 * 4
           self.fc_weight = torch.nn.Parameter(torch.randn(fc_in_features, 10) * 0.1)
           self.fc_bias = torch.nn.Parameter(torch.zeros(10))
           self.params = [self.conv1_weight, self.conv1_bias,
                          self.conv2_weight, self.conv2_bias,
                          self.fc_weight, self.fc_bias]
       def forward(self, x):
           x = conv2d_custom(x, self.conv1_weight, self.conv1_bias)
           x = relu(x)
           x = max_pool2d_custom(x)
           x = conv2d_custom(x, self.conv2_weight, self.conv2_bias)
           x = relu(x)
           x = max_pool2d_custom(x)
           x = flatten(x)
           x = linear_custom(x, self.fc_weight, self.fc_bias)
           return x
   ```
   In the training logs, you will see detailed progress for each epoch, confirming that the custom operations are functioning correctly.

### Custom Transformer Model

The Transformer model in this project is a simplified decoder-only architecture built with custom code. The following components are essential:

1. **Token Embedding:**  
   The embedding layer converts token indices into dense vectors. This is done by indexing a parameter tensor that contains the embedding vectors.
   ```python
   self.embed = torch.nn.Parameter(torch.randn(config.vocab_size, config.dim) * 0.1)
   ```
   This operation is fundamental for transforming discrete tokens into continuous vector space.

2. **Self-Attention Mechanism:**  
   The self-attention function is manually implemented to compute the attention scores between tokens. The process involves:
   - Generating query, key, and value matrices using custom linear operations.
   - Calculating scaled dot-product attention.
   - Applying the softmax function to normalize scores.
   ```python
   def attention(self, Q, K, V):
       d = Q.shape[-1]
       scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
       attn = softmax(scores, dim=-1)
       return attn @ V
   ```
   This snippet emphasizes the crucial role of attention in learning relationships between tokens in a sequence.

3. **Feed-Forward Network with Residual Connections:**  
   After the attention block, the model uses a custom feed-forward network that applies a GELU activation function. The output of the feed-forward network is added to the original input, creating a residual connection.
   ```python
   ff = linear_custom(x, self.ff_weight1, self.ff_bias1)
   ff = gelu(ff)
   ff = linear_custom(ff, self.ff_weight2, self.ff_bias2)
   x = x + ff  # Residual connection
   ```
   This design helps in retaining the original information and stabilizing the gradient flow during training.

4. **Projection Layer:**  
   The final layer projects the output of the Transformer into the vocabulary space to generate logits for each token.
   ```python
   logits = linear_custom(x, self.proj_weight, self.proj_bias)
   ```
   These logits are then used to compute the loss during training.

5. **Training Routine:**  
   Similar to the CNN, the Transformer model uses a custom training loop. The loop processes batches of token sequences, computes the logits using the custom Transformer, and calculates the loss using cross-entropy. The training logs provide clear evidence of convergence over several epochs.

### Summary of the Implementation

- **Custom Code Only:**  
  Every core operation in both models is implemented using low-level tensor operations. No high-level PyTorch layers are used for convolution, activation, pooling, or the attention mechanism. This rigorous constraint ensures that the models operate entirely on custom logic.

- **Limited Data for Rapid Prototyping:**  
  The CNN model is trained on only 200 samples from MNIST instead of the full 60,000. This decision is made to drastically reduce training time, facilitate quick debugging, and ensure that the custom implementations can be validated efficiently. The training logs in the README will show the progress made on this reduced dataset.

- **Detailed Training Logs:**  
  Both models output comprehensive logs that display the training progress for each epoch. These logs are included in the README and serve as evidence that the models are functioning correctly under the strict custom code constraints.


---

This project serves as an educational deep dive into the core mechanics of deep learning, demonstrating that a strong understanding of basic tensor operations is fundamental to building advanced models. The rigorous constraint of using only custom code forces developers to engage deeply with the underlying mathematics and logic of neural networks, making this an invaluable exercise for any deep learning enthusiast.
