"""Trainable input gates — the gated ANN variant.

CORRECTION TO THE DIAGRAMS — read this first:
    The architecture diagrams showed the gates as sigmoid-constrained
    to [0, 1], with g_i^(0) = A_i. That is WRONG for the actual
    implementation, and this module implements the real one:

        g_i^(0)  = sqrt(A_i / mean(A))      (initial gate)
        theta_i  = log(g_i)                  (trainable parameter)
        g_i      = exp(theta_i)              (gate used in forward pass)

    Because exp() is always positive, g_i > 0 always. But there is
    NO upper bound: a gate can grow above 1, and it can shrink below
    1. There is no sigmoid, no logit(A_i), and no [0, 1] constraint
    anywhere in this file. If you describe the gates as sigmoid in
    the report or the viva, that description will be wrong.

Why this parameterisation? A gate multiplies a standardised input,
so a zero or negative gate would flip or kill a feature in a way
that is hard to interpret and hard to train. exp(theta) keeps the
gate positive by construction while still letting the network
learn, per feature, whether to amplify (>1) or suppress (<1) it.

Other facts, all matching the technical document:
* The gated variant uses Xavier W1 with NO row scaling. The gates,
  not the rows, carry the adaptive importance.
* Gates are trained jointly with the network, with the SAME Adam
  optimiser and the SAME learning rate (0.001). There is no
  separate learning rate, no L1 / sparsity penalty, no weight
  decay, and no separate gate optimiser.
* The gate gradient: the gated input is x_tilde = g * x, so
  d(loss)/d(log_gate_i) = g_i * sum_n d(loss)/d(x_tilde[n, i])
  * x[n, i]. The ``dX_gated`` gradient from ann.backward is
  d(loss)/d(x_tilde), and the standardised inputs x are in the
  forward cache, so this is one line — see train_gated().
"""

import numpy as np

from .ann import (
    Adam,
    _copy_params,
    backward,
    forward,
    init_params,
    mse_loss,
    predict,
)
from .config import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
)


def initial_gates(adaptive):
    """Initial gates g_i^(0) = sqrt(A_i / mean(A))."""
    adaptive = np.asarray(adaptive, dtype=float)
    mean_a = adaptive.mean()
    if mean_a <= 0:
        raise ValueError("Adaptive importance must have a positive mean")
    return np.sqrt(adaptive / mean_a)


def gates_from_log_gate(log_gate):
    """Recover the gates: g_i = exp(theta_i). Always positive."""
    return np.exp(log_gate)


def train_gated(X_train, y_train, X_val, y_val, adaptive,
                seed=17, max_epochs=MAX_EPOCHS,
                patience=EARLY_STOPPING_PATIENCE):
    """Train the gated ANN. Returns (best_params, history).

    ``best_params`` is a plain-ANN parameter dict plus a
    ``"log_gate"`` entry (length 7). Pass ``gates_from_log_gate(
    best_params["log_gate"])`` to ``ann.predict`` to use it.
    The training loop mirrors ann._train_loop exactly — same
    full-batch Adam, same early stopping on validation MSE,
    same best-epoch restoration — with the gate gradient added.
    """
    rng = np.random.default_rng(seed)
    params = init_params(rng, n_inputs=X_train.shape[1])
    # Xavier W1, deliberately NOT row-scaled (see module docstring).
    params["log_gate"] = np.log(initial_gates(adaptive))

    adam = Adam(params)
    best_params = _copy_params(params)
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train_mse": [], "val_mse": []}

    for epoch in range(1, max_epochs + 1):
        gates = gates_from_log_gate(params["log_gate"])
        y_hat, cache = forward(params, X_train, gates=gates)
        train_loss = mse_loss(y_hat, y_train)
        grads = backward(params, cache, y_hat, y_train)
        # Gate gradient: dL/d(log_gate_i) = g_i * sum_n dL/d(x_tilde)
        # * x_standardised. cache["X"] is the standardised input
        # BEFORE gating (ann.forward stores it separately from the
        # gated input in cache["X_gated"]); grads["dX_gated"] is
        # dL/d(x_tilde). Using the gated input here would multiply
        # by an extra g_i and give the wrong gradient.
        grads["log_gate"] = gates * np.sum(
            grads["dX_gated"] * cache["X"], axis=0
        )
        # Adam steps over every key in params, including log_gate,
        # with the same learning rate. The extra "dX_gated" key in
        # grads is ignored by Adam (it is not a parameter).
        adam.step(params, grads)

        gates_val = gates_from_log_gate(params["log_gate"])
        y_val_hat, _ = forward(params, X_val, gates=gates_val)
        val_loss = mse_loss(y_val_hat, y_val)
        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_params = _copy_params(params)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    history["best_epoch"] = best_epoch
    history["best_val_mse"] = best_val_loss
    history["n_epochs"] = len(history["train_mse"])
    return best_params, history


def predict_gated(params, X):
    """Predict with a trained gated model (standardised inputs)."""
    gates = gates_from_log_gate(params["log_gate"])
    return predict(params, X, gates=gates)


def final_gates(params):
    """Return the learned gates g_i = exp(theta_i) for reporting."""
    return gates_from_log_gate(params["log_gate"])
