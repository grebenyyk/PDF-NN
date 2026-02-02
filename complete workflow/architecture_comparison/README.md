# Architecture Comparison

This directory contains alternative CNN architectures tested for nuclearity prediction from PDF data.

## Models

### 4-train-model-CSD-with-attention.ipynb
**Architecture:** 2x Conv1D + SeqSelfAttention + Dense layers  
**Accuracy:** ~0.87  
**Characteristics:**
- Includes attention mechanism for feature weighting
- More complex gradient paths
- Less consistent saliency maps
- Higher parameter count

### 4-train-model-CSD-minimal.ipynb
**Architecture:** 1x Conv1D + GlobalAveragePooling + Dense  
**Accuracy:** Lower performance  
**Characteristics:**
- Extremely simplified architecture
- GlobalAveragePooling loses spatial information
- Too simple for PDF analysis
- Noisy, inconsistent saliency maps

## Main Model (in parent directory)

**4-train-model-CSD.ipynb**

**Architecture:** 2x Conv1D + BatchNorm + MaxPooling + Flatten + Dense  
**Accuracy:** ~0.87  
**Selected as optimal because:**
- Same accuracy as attention model
- **Cleanest, most interpretable saliency maps**
- Consistent feature importance within nuclearity classes
- Physically meaningful r-range patterns (metal-metal distances)
- Simpler gradient flow for better interpretability
- Best balance of complexity vs. interpretability

## Key Finding

For PDF-based nuclearity prediction, the no-attention architecture provides the best combination of:
1. Predictive performance
2. Interpretability (clean saliency maps)
3. Chemical relevance (consistent r-range features per nuclearity)

The attention mechanism adds complexity without improving accuracy and produces less interpretable feature importance maps.
