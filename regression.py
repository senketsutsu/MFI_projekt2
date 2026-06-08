"""Regresja wielomianowa z gradient descent – implementacja z notebooka."""

import numpy as np


def normalize_feature(x):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        raise ValueError("Odchylenie standardowe x wynosi 0. Nie można wykonać normalizacji.")
    return (x - mean) / std, mean, std


def create_polynomial_features(x, degree):
    if degree < 0:
        raise ValueError("Stopień wielomianu musi być nieujemny.")

    x = np.asarray(x).reshape(-1, 1)
    X = np.ones((x.shape[0], degree + 1))
    for power in range(1, degree + 1):
        X[:, power] = x[:, 0] ** power
    return X


def predict(X, theta):
    return X @ theta


def compute_loss(X, y, theta):
    m = len(y)
    error = predict(X, theta) - y
    return (1 / (2 * m)) * np.sum(error ** 2)


def compute_gradient(X, y, theta):
    m = len(y)
    error = predict(X, theta) - y
    return (1 / m) * (X.T @ error)


def gradient_descent(
    X,
    y,
    learning_rate=0.01,
    iterations=1000,
    tolerance=1e-8,
):
    theta = np.zeros(X.shape[1])
    loss_history = []

    for i in range(iterations):
        theta -= learning_rate * compute_gradient(X, y, theta)
        loss = compute_loss(X, y, theta)
        loss_history.append(loss)

        if i > 0 and abs(loss_history[-2] - loss_history[-1]) < tolerance:
            break

    return theta, loss_history


def fit_polynomial_regression(x, y, degree, learning_rate=0.0001, iterations=100_000, tolerance=1e-9):
    x_normalized, _, _ = normalize_feature(x)
    X = create_polynomial_features(x_normalized, degree)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        theta, loss_history = gradient_descent(
            X, y, learning_rate=learning_rate, iterations=iterations, tolerance=tolerance
        )

    y_pred = predict(X, theta)
    final_loss = compute_loss(X, y, theta)

    return {
        "theta": theta,
        "y_pred": y_pred,
        "loss_history": loss_history,
        "final_loss": final_loss,
        "iterations": len(loss_history),
    }
