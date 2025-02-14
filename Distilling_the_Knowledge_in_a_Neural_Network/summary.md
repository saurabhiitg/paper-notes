# Distilling the Knowledge in a Neural Network

## Summary

This paper introduces **Knowledge Distillation**, a technique for transferring knowledge from a large, complex model (or an ensemble of models) to a smaller, more efficient model. Unlike previous approaches that directly compress ensemble predictions, this method **softens the probability distributions** of the teacher model using a temperature-scaled softmax. This softened output is used to guide the training of the student model, improving its generalization.

The key insights are:
- A cumbersome model could be an **ensemble of separately trained models** or a **single large model trained with strong regularization** (e.g., dropout).
- Instead of using hard labels, the student model learns from **soft targets**—probability distributions that reveal more about the generalization patterns of the teacher model.
- The distillation process involves raising the **temperature** of the softmax in the teacher model, generating soft targets that the student model is trained to mimic.
- When labeled data is available, the student model can be trained using a **weighted combination of soft and hard targets**, improving performance.

This method allows small models to perform well while being significantly **cheaper to deploy**.
