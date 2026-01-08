r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1. Given $\mathbf{X} \in \mathbb{R}^{N \times D_{in}}$ and $\mathbf{Y} \in \mathbb{R}^{N \times D_{out}}$, the shape is:
$$\text{Shape} = (N, D_{out}, N, D_{in}) = (64, 512, 64, 1024)$$

2. The Jacobian is a diagonal matrix: Because each sample $i$ is indeppendent:
$$\frac{\partial Y_{i, \cdot}}{\partial X_{j, \cdot}} = 0 \quad \text{for} \quad i \neq j$$
Each digonal block is exactly the weight matrix $\mathbf{W}$:
$$\frac{\partial Y_{i, \cdot}}{\partial X_{i, \cdot}} = \mathbf{W}$$

3. Since all blocks are identical, we only need to store the local Jacobian for a single sample.
* **Optimization:** Use the shared weiht matrix instead of the full tensor.
* **New Shape:** $(D_{out}, D_{in}) = (512, 1024)$

4. To calculate $\delta \mathbf{X}$ without materializing the Jacobian:
$$\delta \mathbf{X} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{W}$$
$$\text{Dimensions: } (N \times D_{in}) = (N \times D_{out}) \times (D_{out} \times D_{in})$$

5. Jacobian w.r.t. Weights $\frac{\partial \mathbf{Y}}{\partial \mathbf{W}}$
* **Full Tensor Shape:** $(N, D_{out}, D_{out}, D_{in}) = (64, 512, 512, 1024)$
* **Block Shape:** $(D_{out} \times D_{out})=(512 \times 512)$
"""

part1_q2 = r"""
**Your answer:**


It can be useful in some cases. 
* For example, when the loss acts as a narrow valley, where the slope is very steep on the sides but very flat along the floor. 
* If we use Gradient Descent , it will give very large weight to the slope and little to the floor of the valley, what will cause bouncing across the valley with very slow progress.
* If we have the second derivative, we knoe the curvature, or how quickly the slope is changing .we could know that there is a big slope on the walls and little in the direction of the floor. By this, we can give small steps along the walls and big steps with the direction of the floor, and learn much faster.


"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.05, 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.1,
        0.02,
        0.005,
        0.0002,
        0.001,
    )
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. The results match what we expected to see: 
* The no-dropout learn very fast on the training epochs and get very high accuracy, but is over-fitting very and the tests accuracy stay very low.
* With dropout, the over fitting is lower, so althought the accuracy of the training is lower, the accuracy of the tests is higher (in 0.4 case) or equal (on 0.8).

2. In the low dropout the accuracy of the tests is higher and more stable than the accuracy of the high dropout, especially at the begining. It match what we expected to see, because the high dropout is underfitting, and each time other parameters are learned, so it explain why the accuracy is both low and unstable.



"""

part2_q2 = r"""
**Your answer:**


It can happen, because the loss and the accuracy measure two different things: the loss measure how we sure about the correct class, and the accuracy measure if the correct class is the closest or not.

We can give an example that will demonstrate this situation. Assume there are 2 classes (0,1) each epoch has 2 samples and the true label of both samples is 1.
** epoch 1: **
- sample 1: $[0.49, 0.51]$ : accuracy1 += 1/2, loss1 += 0.673345/2
- sample 2: $[0.49, 0.51]$ : accuracy1 += 1/2, loss1 += 0.673345/2

** epoch 2: **
- sample 1: $[0.01, 0.99]$ : accuracy2 += 1/2 , loss += 0.01005/2
- sample 2: $[0.6, 0.4]$ : accuracy2 += 0/2 (because argmax=0), loss2 += 0.916291/2

We get that the accuracy of epoch 1 (accuracy1) is 1, bigger than accurcy2=0.5. but, also loss1 = 0.673345 > loss2= 0.4631705

It happen because in the first epoch the result was correct, but we were very not sure about them. in the second epoch, one result was wrong (but close) and the other was right and we are very sure about it.


"""

part2_q3 = r"""
**Your answer:**

1. Both algorithms minimize a loss function by updating parameters in the direction of the negative gradient, but they differ in data usage and convergence behavior:

* **Data Usage:** GD uses the entire dataset to compute one update. SGD uses a single random sample (or a mini-batch).
* **Stability:** GD is deterministic and stable. SGD is stochastic, which causes the loss to be unstable but helps in escaping shallow local minima.
* **Computational Efficiency:** GD is very slow and memory-intensive for large datasets. SGD is computationally efficient as it updates parameters frequently with minimal memory overhead. Also, GD updates the parameters once in an epoch, while SGD updates several times.


2. yes, you should use momentom in GD too: momentom help not only by making the convergence smoothier (what helps mostly SGD and not GD), but also in ccelerating progress through flat plateaus and dampening oscillations in narrow valleys where the curvature is ill-conditioned (what helps GD too).

3.1. yes, the approach of summing losses over disjoint batches is **mathematically equivalent** to GD. The total loss $L$ for $N$ samples is the average:

$$L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell_i(\theta)$$

By the linearity of the derivative, the gradient of the sum is the sum of the gradients:

$$\nabla_{\theta} L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} \ell_i(\theta)$$

Splitting the sum into $K$ batches $B_k$:

$$\nabla_{\theta} L(\theta) = \frac{1}{N} \sum_{k=1}^{K} \left( \sum_{i \in B_k} \nabla_{\theta} \ell_i(\theta) \right)$$

3.2. The Out of Memory error occurred because when you perform a forward pass on a batch, the Autograd engine stores all intermediate activations required for the backward pass.
If you wait to call '.backward()' until all batches are processed, the system is forced to keep the computation graphs for the entire dataset in memory simultaneously, and that what make the out of memory error.

3.3. We can call '.backward()' at the end of each batch, and accomulate the result in buffer, that way the memory is freed in the end of each batch and we dont get out of memory error.
"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 2
    hidden_dims = 32
    activation = "relu"
    out_activation = "none"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 0.001
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1.
- optimization error: The gap between the absolute minimum loss and the loss that the model acttualy achieved.
- Generalization error: the gap between the model performence on the given data, and the model performence on a new data. 
- Approximation error: the gap between the best model in the hypothesis class, and the real model that represent the data,

2. 
- optimization error: low. The training loss is very low, and the accuracy is very high, and it's not seem like more epochs could get much better results, or the model is stuck in a local minima. 
- Generalization error: high. there is a gap between the training and the testing, what mean that the model is very good on the known data but has difucalties with new data.
- Approximation error: low. the training got good results, what mean that the model is capable enough to represent the data.
"""

part3_q2 = r"""
**Your answer:**

We will prefer to optimize FPR at the cost of increasing FNR in cases where it is more importent for us not to be wrong about a positive claaification, even at the cost of miss some of the positive results and mark them as negative.
For example, if the model try to predict which spot is good to oil drilling, we will prefer to be very sure about the spots that we drill in, even if we miss some spots. 

We will prefer to optimize FPR at the cost of increasing FNR in cases where it is more importent for us not to miss not to miss right result, even in cost of false positive. 
For example, if the model try to predict a disease, we will want all the suspected to get checked, even if some of them are actuallt healthy.


"""

part3_q3 = r"""
**Your answer:**
1. With the depth fixed, as the width grow the decision boundaries get more smooth and less linear, making the accuracy higher.
2. With the width fixed, as the depth grow the decision boundaries get more curved, what helps the model in the low width cases but may get to over fit with the higher width.
3. The two models get very close results, both on valisation and on test, but their decision boundaries are different. The (4,8) is much curvier than the (1,32). in this case they get similiar result, but with more complexed shape the (4,8) could get mush better results.
4. the threshold selection improved the test results. The test and valisation sets comes from the same distrubution, so when we find the threshold that get balance on the valisation sets, so it match more for the distribution of the validation, and therefore match more to the d
4. the threshold selection improved the test results. The test and valisation sets comes from the same distrubution, so when we find the threshold that get balance on the valisation sets, so it match more for the distribution of the validation, and therefore match more to the 4. the threshold selection improved the test results. The test and valisation sets comes from the same distrubution, so when we find the threshold that get balance on the valisation sets, so it match more for the distribution of the validation, and therefore match more to the distribution of the test.
"""

# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""