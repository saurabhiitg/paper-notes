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
