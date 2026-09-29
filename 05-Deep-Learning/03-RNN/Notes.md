# Recurrent Neural Networks (RNN)

## 1. What is RNN?

RNN stands for **Recurrent Neural Network**.

It is a type of neural network designed to work with **sequential or time-dependent data**.

Examples:

* Text
* Speech
* Time series
* Sensor readings
* Stock data
* Sequential events

The key idea is that an RNN uses information from previous steps when processing the current step.

---

## 2. Why RNN?

Traditional neural networks generally treat inputs independently.

But in sequential data, the order of information matters.

Example:

```text
I am learning Machine Learning
```

The meaning of a word can depend on the words that came before it.

An RNN processes the sequence step by step:

```text
I → am → learning → Machine → Learning
```

and maintains information through its hidden state.

---

## 3. Basic RNN Structure

A simple RNN can be represented as:

```text
Input
  ↓
RNN Cell
  ↓
Hidden State
  ↓
RNN Cell
  ↓
Hidden State
  ↓
Output
```

For a sequence:

```text
x₁ → RNN → h₁
x₂ → RNN → h₂
x₃ → RNN → h₃
x₄ → RNN → h₄
```

Each step receives:

* Current input
* Previous hidden state

---

## 4. RNN Hidden State

The hidden state stores information from previous steps.

Conceptually:

```text
Current Input + Previous Hidden State
                 ↓
              RNN Cell
                 ↓
          New Hidden State
```

This allows the network to carry information through a sequence.

---

## 5. RNN Formula

A simplified RNN equation is:

```text
hₜ = activation(Wₓxₜ + Wₕhₜ₋₁ + b)
```

Where:

* `xₜ` = current input
* `hₜ₋₁` = previous hidden state
* `hₜ` = current hidden state
* `Wₓ` = input weight
* `Wₕ` = recurrent weight
* `b` = bias

The hidden state is then used to produce an output.

---

## 6. RNN Unrolled Through Time

An RNN can be visualized as the same cell repeated across time steps.

```text
x₁        x₂        x₃        x₄
↓         ↓         ↓         ↓
RNN  →   RNN  →   RNN  →   RNN
↓         ↓         ↓         ↓
h₁        h₂        h₃        h₄
```

The RNN cells share the same parameters.

---

## 7. Sequence Example

Consider:

```text
I → love → machine → learning
```

The RNN processes:

```text
Step 1: I
Step 2: love
Step 3: machine
Step 4: learning
```

The hidden state carries information from earlier steps.

---

## 8. Types of RNN Input/Output

RNNs can handle different sequence structures.

### One-to-One

```text
Input → Output
```

Example:

```text
Image → Class
```

This is usually handled by other architectures rather than an RNN.

### One-to-Many

```text
Input → Sequence
```

Example:

```text
Image → Caption
```

### Many-to-One

```text
Sequence → Output
```

Example:

```text
Text → Sentiment
```

### Many-to-Many

```text
Sequence → Sequence
```

Example:

```text
English sentence → Translated sentence
```

---

## 9. RNN for Sentiment Analysis

Suppose we have:

```text
"This movie is excellent"
```

The model processes the words sequentially.

```text
This → movie → is → excellent
```

The final representation can be used to predict:

```text
Positive
```

or:

```text
Negative
```

---

## 10. RNN for Time Series

RNNs can process time-dependent data.

Example:

```text
Temperature

Monday    → 25°C
Tuesday   → 26°C
Wednesday → 27°C
Thursday  → 28°C
```

Previous observations can help the model learn patterns in the sequence.

---

## 11. Vanishing Gradient Problem

A major limitation of basic RNNs is the **vanishing gradient problem**.

During backpropagation through many time steps, gradients can become extremely small.

As a result, the network may have difficulty learning long-term dependencies.

Conceptually:

```text
Earlier Information
       ↓
       ↓
       ↓
Very small gradient
       ↓
Difficult to learn
```

This is one reason architectures such as **LSTM** and **GRU** were developed.

---

## 12. Exploding Gradient Problem

The opposite problem can also occur.

Gradients can become extremely large.

This is called the **exploding gradient problem**.

A common technique to help control this is:

```text
Gradient Clipping
```

---

## 13. RNN Limitations

Basic RNNs have several limitations:

* Difficulty learning long-term dependencies
* Vanishing gradients
* Exploding gradients
* Sequential computation can make training slower
* Difficult to retain information over very long sequences

LSTM and GRU architectures address some of these problems.

---

## 14. RNN vs Feedforward Neural Network

| Feedforward Neural Network               | RNN                                                   |
| ---------------------------------------- | ----------------------------------------------------- |
| Processes inputs without recurrent state | Maintains a hidden state                              |
| Does not naturally model sequence order  | Designed for sequential data                          |
| Previous input is not stored internally  | Previous information can influence current processing |
| Common for tabular tasks                 | Common for sequence tasks                             |

---

## 15. RNN vs CNN

| CNN                            | RNN                                   |
| ------------------------------ | ------------------------------------- |
| Commonly used for spatial data | Commonly used for sequential data     |
| Learns local spatial patterns  | Learns patterns across sequence steps |
| Common in computer vision      | Common in sequence modeling           |
| Uses convolution               | Uses recurrent connections            |

---

## 16. RNN Training

RNNs are trained using a technique called:

**Backpropagation Through Time (BPTT)**

The basic process is:

```text
Input Sequence
      ↓
Forward Pass
      ↓
Prediction
      ↓
Loss
      ↓
Backpropagation Through Time
      ↓
Gradient Calculation
      ↓
Parameter Update
```

---

## 17. Backpropagation Through Time

In BPTT, the RNN is conceptually unrolled across its time steps.

Example:

```text
x₁ → RNN → x₂ → RNN → x₃ → RNN
              ↓
          Backpropagation
```

The gradients are calculated through the sequence.

This can lead to vanishing or exploding gradients when sequences are long.

---

## 18. LSTM

LSTM stands for:

**Long Short-Term Memory**

LSTM is a special type of recurrent neural network designed to better handle long-term dependencies.

It uses gates to control information flow.

Main gates:

* Forget gate
* Input gate
* Output gate

LSTM is covered in the next topic.

---

## 19. GRU

GRU stands for:

**Gated Recurrent Unit**

GRU is another recurrent architecture designed to handle long-term dependencies.

It is generally simpler than LSTM because it uses fewer gates.

GRU is also covered in the next topic.

---

## 20. Applications of RNN

RNNs have been used for:

### Natural Language Processing

* Text classification
* Sentiment analysis
* Language modeling
* Text generation

### Speech

* Speech recognition
* Sequence processing

### Time Series

* Forecasting
* Sensor data analysis

### Other Sequential Data

* Event prediction
* Activity recognition

Modern applications often use architectures such as Transformers instead of basic RNNs for many NLP tasks.

---

## 21. Simple RNN Example

Suppose we want to predict the next value in a sequence:

```text
10 → 12 → 14 → 16 → ?
```

The RNN processes previous values and learns patterns in the sequence.

It may produce:

```text
Prediction = 18
```

The prediction is then compared with the actual value using a loss function.

---

## 22. RNN with PyTorch

A simple RNN layer can be created using PyTorch:

```python
import torch
import torch.nn as nn

rnn = nn.RNN(
    input_size=10,
    hidden_size=20,
    batch_first=True
)
```

Where:

* `input_size=10` → number of features at each time step
* `hidden_size=20` → number of hidden-state features
* `batch_first=True` → input shape starts with batch size

Typical input shape:

```text
(batch_size, sequence_length, input_size)
```

---

## 23. RNN with Keras

A simple RNN layer can be created using Keras:

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN

model = Sequential([
    SimpleRNN(32, input_shape=(10, 5))
])
```

Here:

```text
10 → sequence length
5  → features per time step
32 → number of RNN units
```

---

## 24. Important Terms

| Term               | Meaning                                |
| ------------------ | -------------------------------------- |
| RNN                | Recurrent Neural Network               |
| Sequence           | Ordered set of data                    |
| Hidden State       | Stores information from previous steps |
| Time Step          | One position in a sequence             |
| BPTT               | Backpropagation Through Time           |
| Vanishing Gradient | Gradients become very small            |
| Exploding Gradient | Gradients become very large            |
| LSTM               | Long Short-Term Memory                 |
| GRU                | Gated Recurrent Unit                   |

---

## 25. Interview Questions

### Q1. What is RNN?

RNN is a neural network designed to process sequential data using recurrent connections and a hidden state.

### Q2. Why is hidden state used in RNN?

The hidden state carries information from previous time steps to the current time step.

### Q3. What is the vanishing gradient problem?

It occurs when gradients become very small during training, making it difficult for the network to learn long-term dependencies.

### Q4. What is BPTT?

BPTT stands for Backpropagation Through Time. It is used to train recurrent neural networks by propagating errors through the unrolled sequence.

### Q5. What is the difference between RNN and LSTM?

LSTM is a specialized recurrent architecture that uses gates to better control information flow and handle long-term dependencies.

### Q6. What is GRU?

GRU is a gated recurrent architecture designed to handle long-term dependencies with a simpler structure than LSTM.

### Q7. Where are RNNs used?

RNNs can be used for sequence tasks such as text processing, speech, and time-series data.

---

## 26. Summary

The main idea of an RNN is:

```text
Current Input
      +
Previous Hidden State
      ↓
   RNN Cell
      ↓
New Hidden State
      ↓
Next Time Step
```

RNNs are designed for sequential data, but basic RNNs can struggle with long-term dependencies because of vanishing and exploding gradients.

This led to improved architectures such as:

```text
RNN
 ↓
LSTM
 ↓
GRU
```

