# LSTM and GRU

## 1. Introduction

LSTM and GRU are advanced types of **Recurrent Neural Networks (RNNs)**.

They were designed to handle problems that basic RNNs can face when learning information over long sequences.

The main problem is:

```text
RNN
 ↓
Long Sequence
 ↓
Vanishing / Exploding Gradients
 ↓
Difficulty Learning Long-Term Dependencies
```

LSTM and GRU use **gates** to control which information should be kept, updated, or discarded.

---

# 2. LSTM

LSTM stands for:

**Long Short-Term Memory**

LSTM is a type of RNN designed to remember useful information for longer periods.

It was introduced to address the limitations of traditional RNNs.

---

## 3. Why LSTM?

Consider the sentence:

```text
"I grew up in India. I speak fluent ______."
```

To predict the missing word, the model may need to remember information from earlier in the sentence.

A basic RNN may have difficulty maintaining this information across many time steps.

LSTM is designed to handle such long-term dependencies more effectively.

---

## 4. LSTM Architecture

An LSTM cell contains:

* Cell State
* Hidden State
* Forget Gate
* Input Gate
* Output Gate

Basic idea:

```text
              ┌───────────────┐
Previous ────>│   LSTM Cell   │────> New Hidden State
Hidden State  │               │
              │  Gates + Cell  │
              │     State      │
              └───────────────┘
                     ↑
                Current Input
```

---

# 5. Cell State

The cell state acts like a memory path.

It carries information across time steps.

```text
Cell State
──────────────────────────────>
       ↑          ↑          ↑
      Gate       Gate       Gate
```

The gates control what information is added or removed.

---

# 6. Hidden State

The hidden state contains the information that the LSTM exposes as its current output representation.

At each time step:

```text
Current Input
      +
Previous Hidden State
      +
Previous Cell State
      ↓
   LSTM Cell
      ↓
New Hidden State
      +
New Cell State
```

---

# 7. Forget Gate

The forget gate decides what information from the previous cell state should be removed.

It uses a sigmoid function.

Conceptually:

```text
Previous Cell State
        ↓
   Forget Gate
        ↓
What should be forgotten?
```

The output is between:

```text
0 and 1
```

Interpretation:

```text
0 → Forget
1 → Keep
```

Formula:

```text
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
```

---

# 8. Input Gate

The input gate decides what new information should be stored in the cell state.

It has two main parts:

1. Decide what information to update.
2. Create candidate information.

Conceptually:

```text
Current Input
      ↓
Input Gate
      ↓
New Information
      ↓
Cell State
```

---

# 9. Candidate Cell State

The LSTM creates candidate information that may be added to the cell state.

A common formulation uses `tanh`:

```text
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)
```

The input gate controls how much of this candidate information is actually added.

---

# 10. Updating the Cell State

The old cell state is updated using:

```text
Old Memory
     ↓
Forget Gate
     ↓
+
New Candidate Information
     ↓
Input Gate
     ↓
New Memory
```

Formula:

```text
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ
```

Where:

* `Cₜ` = new cell state
* `Cₜ₋₁` = previous cell state
* `fₜ` = forget gate
* `iₜ` = input gate
* `C̃ₜ` = candidate cell state

---

# 11. Output Gate

The output gate decides what information from the updated cell state should become the new hidden state.

Conceptually:

```text
Updated Cell State
       ↓
  Output Gate
       ↓
New Hidden State
```

Formula:

```text
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
```

The new hidden state is commonly:

```text
hₜ = oₜ * tanh(Cₜ)
```

---

# 12. Complete LSTM Flow

The simplified LSTM process is:

```text
Previous Hidden State
        +
Current Input
        ↓
   Forget Gate
        ↓
Remove unnecessary information
        ↓
   Input Gate
        ↓
Add useful new information
        ↓
 Update Cell State
        ↓
   Output Gate
        ↓
 New Hidden State
```

---

# 13. LSTM Gates

| Gate        | Main Purpose                           |
| ----------- | -------------------------------------- |
| Forget Gate | Decides what old information to remove |
| Input Gate  | Decides what new information to store  |
| Output Gate | Decides what information to output     |

---

# 14. LSTM Example

Consider:

```text
"I studied Python for two years, so I am comfortable with ______."
```

The model needs to retain useful information from earlier words.

LSTM's memory mechanism helps it carry relevant information across multiple time steps.

---

# 15. LSTM Applications

LSTM can be used for:

### NLP

* Text classification
* Sentiment analysis
* Language modeling
* Sequence generation

### Time Series

* Forecasting
* Sensor data analysis
* Demand prediction

### Speech

* Speech recognition
* Sequence modeling

### Other Sequential Data

* Activity recognition
* Event prediction

---

# 16. GRU

GRU stands for:

**Gated Recurrent Unit**

GRU is another type of gated recurrent neural network.

It was designed to provide a simpler alternative to LSTM.

---

# 17. Why GRU?

GRU addresses long-term dependency problems using gates, but its architecture is simpler than LSTM.

Basic idea:

```text
Input
  +
Previous Hidden State
  ↓
 GRU Cell
  ↓
New Hidden State
```

Unlike LSTM, GRU does not maintain a separate cell state.

---

# 18. GRU Gates

A standard GRU mainly uses two gates:

* Update Gate
* Reset Gate

---

# 19. Update Gate

The update gate determines how much previous information should be kept and how much new information should be used.

Conceptually:

```text
Previous Information
        +
New Information
        ↓
   Update Gate
        ↓
New Hidden State
```

A common formulation is:

```text
zₜ = σ(Wz · [hₜ₋₁, xₜ])
```

---

# 20. Reset Gate

The reset gate determines how much previous information should be ignored when creating the candidate hidden state.

Formula:

```text
rₜ = σ(Wr · [hₜ₋₁, xₜ])
```

Conceptually:

```text
Previous Hidden State
        ↓
    Reset Gate
        ↓
How much old information should be used?
```

---

# 21. Candidate Hidden State

The GRU creates a candidate hidden state using the current input and reset-controlled previous state.

A common formulation is:

```text
h̃ₜ = tanh(Wh · [rₜ * hₜ₋₁, xₜ])
```

---

# 22. GRU Hidden State Update

The update gate combines the previous hidden state and candidate hidden state.

A common formulation is:

```text
hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ
```

Different implementations may use an equivalent convention for the update gate.

---

# 23. LSTM vs GRU

| LSTM                                             | GRU                                 |
| ------------------------------------------------ | ----------------------------------- |
| Uses forget, input and output gates              | Uses update and reset gates         |
| Has hidden state and cell state                  | Uses hidden state only              |
| More complex architecture                        | Simpler architecture                |
| More parameters                                  | Fewer parameters                    |
| Can be useful for complex long-term dependencies | Often faster and more lightweight   |
| Can require more computation                     | Generally requires less computation |

Neither architecture is universally better. The appropriate choice depends on the dataset, task, and computational requirements.

---

# 24. RNN vs LSTM vs GRU

| Feature                | RNN       | LSTM            | GRU                     |
| ---------------------- | --------- | --------------- | ----------------------- |
| Hidden State           | Yes       | Yes             | Yes                     |
| Cell State             | No        | Yes             | No                      |
| Gates                  | No        | 3 main gates    | 2 main gates            |
| Long-term dependencies | Difficult | Better handling | Better handling         |
| Architecture           | Simple    | More complex    | Simpler than LSTM       |
| Parameters             | Fewer     | More            | Usually fewer than LSTM |

---

# 25. LSTM Example with PyTorch

A basic LSTM layer:

```python
import torch.nn as nn

lstm = nn.LSTM(
    input_size=10,
    hidden_size=32,
    batch_first=True
)
```

Typical input shape:

```text
(batch_size, sequence_length, input_size)
```

For example:

```text
32 samples
10 time steps
5 features
```

Shape:

```text
(32, 10, 5)
```

---

# 26. GRU Example with PyTorch

A basic GRU layer:

```python
import torch.nn as nn

gru = nn.GRU(
    input_size=10,
    hidden_size=32,
    batch_first=True
)
```

The input format is similar to LSTM:

```text
(batch_size, sequence_length, input_size)
```

---

# 27. LSTM Example with Keras

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM

model = Sequential([
    LSTM(32, input_shape=(10, 5))
])
```

Here:

```text
10 → sequence length
5  → features per time step
32 → LSTM units
```

---

# 28. GRU Example with Keras

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU

model = Sequential([
    GRU(32, input_shape=(10, 5))
])
```

---

# 29. LSTM and GRU Training

Both LSTM and GRU are trained using gradient-based optimization.

General process:

```text
Input Sequence
      ↓
LSTM / GRU
      ↓
Prediction
      ↓
Loss
      ↓
Backpropagation Through Time
      ↓
Gradients
      ↓
Optimizer
      ↓
Updated Parameters
```

---

# 30. Advantages of LSTM

* Handles long-term dependencies better than basic RNNs
* Uses gates to control information
* Useful for sequential data
* Can retain important information over many time steps

---

# 31. Limitations of LSTM

* More parameters than basic RNN
* More computationally expensive
* Training can be slower
* Sequential processing limits parallelism compared with some modern architectures

---

# 32. Advantages of GRU

* Simpler than LSTM
* Fewer parameters
* Often faster to train
* Handles long-term dependencies better than basic RNN
* Useful when a lighter recurrent architecture is desired

---

# 33. Limitations of GRU

* May not perform best on every sequence problem
* Still uses sequential computation
* Long sequences can remain challenging
* Performance depends on the task and dataset

---

# 34. LSTM and GRU Applications

Both can be used for:

```text
Text
 ↓
Sentiment Analysis
 ↓
Language Modeling
 ↓
Sequence Classification
```

and:

```text
Sensor Data
 ↓
Time Series
 ↓
Forecasting
```

They can also be used for speech and other sequential data.

---

# 35. Important Terms

| Term         | Meaning                                        |
| ------------ | ---------------------------------------------- |
| LSTM         | Long Short-Term Memory                         |
| GRU          | Gated Recurrent Unit                           |
| Cell State   | LSTM memory pathway                            |
| Hidden State | Current representation/output state            |
| Forget Gate  | Controls removal of old information            |
| Input Gate   | Controls addition of new information           |
| Output Gate  | Controls LSTM output                           |
| Update Gate  | Controls GRU information update                |
| Reset Gate   | Controls how much previous information is used |
| BPTT         | Backpropagation Through Time                   |

---

# 36. Interview Questions

### Q1. What is LSTM?

LSTM is a type of recurrent neural network designed to handle long-term dependencies using a memory cell and gates.

### Q2. Why was LSTM introduced?

LSTM was introduced to address limitations of traditional RNNs, particularly difficulty learning long-term dependencies.

### Q3. What are the main LSTM gates?

The main gates are:

* Forget gate
* Input gate
* Output gate

### Q4. What is a cell state?

The cell state is an LSTM memory pathway that carries information across time steps.

### Q5. What is GRU?

GRU is a gated recurrent neural network architecture that handles long-term dependencies using update and reset gates.

### Q6. What is the main difference between LSTM and GRU?

LSTM has a separate cell state and three main gates, while GRU uses a hidden state with two main gates.

### Q7. Which has fewer parameters, LSTM or GRU?

GRU generally has fewer parameters because its architecture is simpler.

### Q8. Are LSTM and GRU always better than RNN?

Not necessarily. They can handle long-term dependencies better, but the appropriate architecture depends on the problem, data, and computational requirements.

---

# 37. Summary

Basic RNN:

```text
Input
 ↓
RNN
 ↓
Hidden State
```

LSTM:

```text
Input + Hidden State + Cell State
              ↓
         LSTM Gates
              ↓
      New Hidden State
      New Cell State
```

GRU:

```text
Input + Previous Hidden State
              ↓
          GRU Gates
              ↓
      New Hidden State
```

The main idea is:

```text
RNN
 ↓
Difficulty with long-term dependencies
 ↓
LSTM / GRU
 ↓
Gated information flow
```

