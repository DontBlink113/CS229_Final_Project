# Online Handwritten Chinese Character Recognition

Classifying online handwritten Chinese characters (500 classes), with a focus on robustness to
second-language-learner errors (omitted, reordered, reversed strokes). Compares a
logistic-regression baseline against a CNN on identical flow-field features. CS229 final project.

> Not runnable as-is — depends on the CASIA-OLHWDB dataset (licensed, not committed). See CS229_Final_Project.pdf
> & final_project_poster.png, or view the graphs * code in the casia_ML.ipynb

## Results

Test accuracy, original vs. augmented (learner-style errors):

| Model               | Test Acc | Aug. |
|---------------------|:--------:|:----:|
| Logistic Regression | 86.3%    | 79.0% |
| CNN (8×8×8 map)      | 92.5%    | 87.8% |

Same 512-d features for both, so the gap measures the value of spatial structure. The CNN also
degrades less under augmentation, suggesting it learns more structural patterns.

Requires CASIA-OLHWDB (Pot1.0/1.1). Stack: Python, PyTorch, NumPy.
