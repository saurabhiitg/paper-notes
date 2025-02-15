# Distilling the Knowledge in a Neural Network

## Why "Distillation"?

- The term **distillation** is borrowed from physics and chemistry, where distillation refers to extracting essential components from a mixture.
- Here, it refers to extracting useful knowledge from a large model (or ensemble) into a much smaller model while preserving its predictive power.

## Soft Targets

- **Soft targets** refer to the probability distribution over all possible classes instead of just the hard label.
- Example: If a teacher model classifies an image as `dog` with 80% confidence, `wolf` with 15%, and `fox` with 5%, this distribution carries richer information than just labeling it as `dog`.
- The student model, when trained on these **softer probabilities**, can generalize better than when trained only on hard labels.

## Temperature in Softmax

- A neural network’s final layer typically applies a **softmax function**:

  \[
  q_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
  \]

  where:
  - \( T \) is the **temperature**, normally set to 1.
  - A higher \( T \) produces a **smoother** probability distribution by reducing the difference between the largest and smaller logits.

- In distillation, the teacher model uses a high temperature to **produce soft targets**, which the student model then learns from.

## Two-Stage Learning

1. **Generate Soft Targets**: The cumbersome model generates soft probabilities with a high-temperature softmax.
2. **Train the Student Model**: The student is trained using:
   - **Cross-entropy with soft targets** (computed using the same high temperature)
   - **Cross-entropy with hard labels** (if available)
   - A **weighted combination** of these two objectives often works best.

## Gradient Scaling

- The magnitudes of the gradients from the soft targets scale as \( \frac{1}{T^2} \), so they must be multiplied by \( T^2 \) when combining hard and soft targets.

---

## Derivation: Cross-Entropy Gradient in Distillation

Each case in the transfer set contributes a **cross-entropy gradient**, \( \frac{\partial C}{\partial z_i} \), with respect to each logit \( z_i \) of the student model. If the teacher model has logits \( v_i \), which produce soft targets \( p_i \), and the transfer training is done at temperature \( T \), the gradient is:

\[
\frac{\partial C}{\partial z_i} = \frac{1}{T} \left( q_i - p_i \right)
\]

where:
- \( p_i = \frac{e^{v_i/T}}{\sum_j e^{v_j/T}} \) (soft targets from the teacher model)
- \( q_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} \) (student model’s probability distribution)

Using the **Taylor expansion** of \( e^x \) for small \( x \), we approximate:

\[
e^{z_i/T} \approx 1 + \frac{z_i}{T}, \quad e^{v_i/T} \approx 1 + \frac{v_i}{T}
\]

So, for large \( T \),

\[
p_i \approx \frac{1}{n} + \frac{1}{T} (v_i - \frac{1}{n} \sum_j v_j), \quad q_i \approx \frac{1}{n} + \frac{1}{T} (z_i - \frac{1}{n} \sum_j z_j)
\]

Thus,

\[
q_i - p_i \approx \frac{1}{T} \left( (z_i - \frac{1}{n} \sum_j z_j) - (v_i - \frac{1}{n} \sum_j v_j) \right).
\]

Finally,

\[
\frac{\partial C}{\partial z_i} \approx \frac{1}{T^2} \left( z_i - v_i - \frac{1}{n} \sum_j (z_j - v_j) \right).
\]

This shows that, for high \( T \), the training process is **smoother**, helping the student model learn effectively from the teacher model.

---

## Key Takeaways

- **Knowledge distillation** extracts information from a large model and compresses it into a smaller model.
- **Soft targets** provide richer training signals than hard labels alone.
- **Temperature scaling** in softmax helps produce softer probability distributions for better generalization.
- A **weighted combination** of soft and hard targets improves student performance.
- **Gradient scaling** ensures proper learning dynamics when both objectives are used.

This technique is widely used in **model compression, transfer learning, and efficient deployment** of deep networks in real-world applications.
