# Step-by-Step Explanation of LSTM (Long Short-Term Memory)

LSTMs are a type of recurrent neural network (RNN) designed to handle long-term dependencies in sequential data. They achieve this by introducing a memory cell and three special gates: **Forget Gate, Input Gate, and Output Gate**. Let's go step by step.

## 1. Forget Gate: What Should Be Remembered or Forgotten?
The first step in an LSTM is to decide what information to keep or discard from the cell state. This decision is made by the **forget gate**, a sigmoid layer that looks at the **previous hidden state** (\(h_{t-1}\)) and **current input** (\(x_t\)) and outputs a value between 0 and 1 for each element in the cell state (\(C_{t-1}\)).

- **1 means "keep everything"**.
- **0 means "completely forget"**.

### Formula:
\[
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
\]
where:
- \( W_f \) and \( b_f \) are the weight matrix and bias for the forget gate.
- \( \sigma \) is the sigmoid activation function.

**Example (Language Model):**  
If the model is processing a sentence, the cell state may store the **gender of the subject**. When a new subject appears, the forget gate decides to forget the old gender.

---

## 2. Input Gate: What New Information to Store?
Next, the model decides what new information should be added to the memory. This consists of two parts:

1. **The input gate (sigmoid layer)** decides **which values to update**.
2. **A tanh layer** generates candidate values (\(\tilde{C}_t\)) to be added to the state.

### Formula:
\[
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
\]
\[
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
\]

**Example (Language Model):**  
The model adds the **new subject’s gender** to the memory to replace the old one.

---

## 3. Updating the Cell State
Now, the **cell state (\(C_t\))** is updated based on the forget and input gates:

\[
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
\]

- **Old state (\(C_{t-1}\)) is multiplied by \(f_t\)** → Forget unnecessary information.
- **New candidate values (\(\tilde{C}_t\)) are scaled by \(i_t\) and added** → Store new relevant information.

**Example (Language Model):**  
The model removes the old subject’s gender from the memory and adds the new one.

---

## 4. Output Gate: What Should Be Output?
Finally, the output is generated based on the new cell state. The output gate determines **which part of the cell state to use** for the hidden state (\(h_t\)).

1. A sigmoid layer selects parts of the cell state to output.
2. The cell state is passed through a tanh activation function.
3. The result is multiplied by the output gate’s decision.

### Formula:
\[
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
\]
\[
h_t = o_t * \tanh(C_t)
\]

**Example (Language Model):**  
If the next word should be a **verb**, the model outputs information about singular/plural form to ensure correct conjugation.

---

## Summary of LSTM Flow
1. **Forget gate**: Decides what to forget from the previous state.
2. **Input gate**: Decides what new information to add.
3. **Cell state update**: Combines forget and input decisions.
4. **Output gate**: Decides what information to pass to the next step.

This structure allows LSTMs to retain important long-term dependencies while discarding irrelevant details, making them powerful for tasks like text generation, speech recognition, and time-series forecasting.
