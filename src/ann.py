"""Custom NumPy ANN: 7-16-8-1, manual backpropagation, custom Adam.

No PyTorch / TensorFlow is used for the ANN. That is deliberate:
the project is about *initialisation*, and a hand-written network
makes the initial weights — the thing being studied — completely
visible, instead of hiding them inside a framework.

Architecture:  7 -> 16 (ReLU) -> 8 (ReLU) -> 1 (linear)
Loss:          mean squared error, on the standardised target.
Optimiser:     Adam, lr 0.001, beta1 0.9, beta2 0.999, eps 1e-8.
Training:      full-batch (the whole training set every epoch),
               max 1,800 epochs, early stopping with patience 140
               on validation MSE. The best-epoch parameters are
               restored, so the returned model is the best one seen,
               not the last one.
               No dropout, no batch normalisation, no weight decay.

Initialisation — the heart of the project:
* Xavier (Glorot uniform): each weight drawn from
  Uniform(-limit, +limit) with limit = sqrt(6 / (fan_in+fan_out)),
  biases zero. This is the "standard" method everything is
  compared against.
* Domain-only / SHAP-only / Adaptive: start from Xavier, then scale
  each row i of W1 by r_i = sqrt(I_i / mean(I)), where I is D, S or
  A respectively, then restore the Frobenius norm of W1 to the
  Xavier value. Scaling a row makes that input start with a larger
  (or smaller) influence; restoring the norm keeps the overall
  weight scale unchanged, so the comparison is about the
  *distribution* of importance across inputs, not about a bigger
  or smaller network overall.
* Gated: Xavier W1 with NO row scaling, plus trainable input gates
  — see gated.py. The gates, not the rows, carry the importance.

``train_ann`` takes the importance vector and a method name and
returns the trained parameters plus a small history dict. Keeping
the parameters in a plain dict (``{"W1": ..., "b1": ..., ...}``)
makes them easy to save, inspect, and restore for early stopping.
"""

import numpy as np

from .config import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    EARLY_STOPPING_PATIENCE,
    HIDDEN_1,
    HIDDEN_2,
    LEARNING_RATE,
    MAX_EPOCHS,
)


# ------------------------------------------------------------ utilities
def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    """Derivative of ReLU: 1 where x > 0, else 0."""
    return (x > 0).astype(float)


def xavier_uniform(rng, fan_in, fan_out):
    """Glorot uniform initialisation for a (fan_in, fan_out) matrix."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def scale_rows_by_importance(W1, importance):
    """Scale W1 rows by sqrt(I_i / mean(I)), restoring Frobenius norm.

    ``importance`` is a length-7 vector (D, S or A) in FEATURES order.
    Returns a new W1 with the same Frobenius norm as the input, but
    with rows for more important inputs made larger.
    """
    importance = np.asarray(importance, dtype=float)
    if np.any(importance < 0):
        raise ValueError("Importance scores must be non-negative")
    mean_importance = importance.mean()
    if mean_importance <= 0:
        raise ValueError("Importance scores sum to zero")
    row_scale = np.sqrt(importance / mean_importance)
    scaled = W1 * row_scale[:, np.newaxis]
    # Restore the original Frobenius norm (eps avoids /0 if the
    # scaled matrix were ever all zeros, which it cannot be here).
    norm_before = np.linalg.norm(W1)
    norm_after = np.linalg.norm(scaled)
    return scaled * (norm_before / (norm_after + 1e-12))


# ------------------------------------------------------------ the network
def init_params(rng, n_inputs=7):
    """Xavier-initialised parameters for a 7-16-8-1 network."""
    return {
        "W1": xavier_uniform(rng, n_inputs, HIDDEN_1),
        "b1": np.zeros(HIDDEN_1),
        "W2": xavier_uniform(rng, HIDDEN_1, HIDDEN_2),
        "b2": np.zeros(HIDDEN_2),
        "W3": xavier_uniform(rng, HIDDEN_2, 1),
        "b3": np.zeros(1),
    }


def apply_importance_init(params, importance):
    """Apply row-scaling initialisation to W1 (non-gated methods)."""
    params = {k: v.copy() for k, v in params.items()}
    params["W1"] = scale_rows_by_importance(params["W1"], importance)
    return params


def forward(params, X, gates=None):
    """Forward pass. Returns (prediction, cache for backprop).

    ``gates`` (length 7, or None) multiplies the inputs element-wise
    before the first layer — used only by the gated variant.
    Prediction has shape (n, 1).
    """
    # Keep the pre-gating input as cache["X"] and the gated input
    # as cache["X_gated"]: backward() needs the gated input for
    # the W1 gradient, while gated.py needs the pre-gating input
    # for the log-gate gradient. Storing both (instead of
    # overwriting X) is what makes both gradients correct.
    X_orig = X
    if gates is not None:
        X = X * gates
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    y_hat = a2 @ params["W3"] + params["b3"]
    cache = {"X": X_orig, "X_gated": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return y_hat, cache


def mse_loss(y_hat, y):
    return float(np.mean((y_hat - y) ** 2))


def backward(params, cache, y_hat, y):
    """Gradients of MSE w.r.t. every parameter (full-batch).

    Returns a dict with the same keys as ``params``. If gates were
    used, the gradient w.r.t. the (standardised) gated inputs is
    returned under the key ``"dX_gated"`` so gated.py can turn it
    into a gradient w.r.t. log-gate.
    """
    n = y.shape[0]
    # The gradient w.r.t. W1 uses the GATED input (the input that
    # actually entered the first layer), not the pre-gating input.
    X = cache["X_gated"]
    # d(loss)/d(y_hat) for MSE = 2 * (y_hat - y) / n.
    d_y_hat = 2.0 * (y_hat - y) / n
    grads = {}
    grads["W3"] = cache["a2"].T @ d_y_hat
    grads["b3"] = d_y_hat.sum(axis=0)
    d_a2 = d_y_hat @ params["W3"].T
    d_z2 = d_a2 * relu_grad(cache["z2"])
    grads["W2"] = cache["a1"].T @ d_z2
    grads["b2"] = d_z2.sum(axis=0)
    d_a1 = d_z2 @ params["W2"].T
    d_z1 = d_a1 * relu_grad(cache["z1"])
    grads["W1"] = X.T @ d_z1
    grads["b1"] = d_z1.sum(axis=0)
    grads["dX_gated"] = d_z1 @ params["W1"].T
    return grads


class Adam:
    """Adam optimiser over a dict of parameter arrays.

    One Adam state (m, v) per parameter, one shared step counter.
    This is the standard Adam update with bias correction; the
    same instance is used for the network weights and, in the
    gated variant, for log-gate as well — there is deliberately
    no separate learning rate for the gates.
    """

    def __init__(self, params, lr=LEARNING_RATE,
                 beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=ADAM_EPS):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        """Apply one Adam update in place; returns the params dict."""
        self.t += 1
        for key in params:
            grad = grads[key]
            self.m[key] = (
                self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
            )
            self.v[key] = (
                self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad ** 2)
            )
            m_hat = self.m[key] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1.0 - self.beta2 ** self.t)
            params[key] = params[key] - self.lr * m_hat / (
                np.sqrt(v_hat) + self.eps
            )
        return params


def _copy_params(params):
    return {k: v.copy() for k, v in params.items()}


def train_ann(X_train, y_train, X_val, y_val, importance=None,
              method="xavier", seed=17, max_epochs=MAX_EPOCHS,
              patience=EARLY_STOPPING_PATIENCE):
    """Train one ANN and return (best_params, history).

    ``importance`` is the D / S / A vector for the non-gated methods;
    it is ignored for ``method="xavier"``. For the gated variant use
    ``gated.train_gated`` instead — it has its own loop, identical
    in structure, because the gates add one extra parameter and one
    extra gradient that this loop does not know about.

    ``y_train`` / ``y_val`` must already be standardised (the caller
    standardises the target with a scaler fitted on train only).
    History contains the per-epoch train/val MSE and the epoch of
    the best validation loss.
    """
    rng = np.random.default_rng(seed)
    params = init_params(rng, n_inputs=X_train.shape[1])
    if method in ("domain", "shap", "adaptive"):
        if importance is None:
            raise ValueError(f"Method {method!r} needs an importance vector")
        params = apply_importance_init(params, importance)
    elif method != "xavier":
        raise ValueError(f"Unknown non-gated method {method!r}")
    return _train_loop(params, X_train, y_train, X_val, y_val,
                       max_epochs=max_epochs, patience=patience)


def _train_loop(params, X_train, y_train, X_val, y_val,
                max_epochs, patience):
    """Full-batch training loop for the plain (non-gated) ANN.

    The loop returns the best-epoch parameters, not the last ones.
    """
    adam = Adam(params)
    best_params = _copy_params(params)
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train_mse": [], "val_mse": []}

    for epoch in range(1, max_epochs + 1):
        y_hat, cache = forward(params, X_train)
        train_loss = mse_loss(y_hat, y_train)
        grads = backward(params, cache, y_hat, y_train)
        adam.step(params, grads)

        y_val_hat, _ = forward(params, X_val)
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


def predict(params, X, gates=None):
    """Return predictions of shape (n, 1) for standardised inputs."""
    y_hat, _ = forward(params, X, gates=gates)
    return y_hat
