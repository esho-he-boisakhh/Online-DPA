import numpy as np

class LinearRegressionOnline:
    #init
    def __init__(self, feature_dim: int):
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")

        self.feature_dim = feature_dim
        self.theta = np.zeros(feature_dim, dtype=np.float64)

    #pred
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            if X.shape[0] != self.feature_dim:
                raise ValueError("Input dimension mismatch")
            return float(np.dot(X, self.theta))

        elif X.ndim == 2:
            if X.shape[1] != self.feature_dim:
                raise ValueError("Input dimension mismatch")
            return X @ self.theta

        else:
            raise ValueError("Input must be 1D or 2D array")

    #mse
    def compute_mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error over dataset.

        MSE = (1/n) * Σ (y_hat - y)^2
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        y_hat = self.predict(X)
        errors = y_hat - y
        return float(np.mean(errors ** 2))

    #single sample grad
    def gradient_single(self, x: np.ndarray, y: float) -> np.ndarray:
        if x.shape[0] != self.feature_dim:
            raise ValueError("Input dimension mismatch")

        y_hat = np.dot(x, self.theta)
        error = y_hat - y

        grad = 2 * error * x
        return grad.astype(np.float64)

    #mini batch grad
    def gradient_batch(self,
                       X_batch: np.ndarray,
                       y_batch: np.ndarray) -> np.ndarray:
        if X_batch.shape[0] != y_batch.shape[0]:
            raise ValueError("Batch size mismatch")

        y_hat = self.predict(X_batch)
        errors = y_hat - y_batch

        grad = 2 * (X_batch.T @ errors) / X_batch.shape[0]
        return grad.astype(np.float64)
    
    #ombgd param update
    def update_parameters(self,
                        gradient: np.ndarray,
                        learning_rate: float):
        
        if gradient.shape[0] != self.feature_dim:
            raise ValueError("Gradient dimension mismatch")

        self.theta = self.theta - learning_rate * gradient

    #jacobian
    def jacobian_theta_update(self,
                              x: np.ndarray,
                              learning_rate: float) -> np.ndarray:
        if x.shape[0] != self.feature_dim:
            raise ValueError("Input dimension mismatch")

        identity = np.eye(self.feature_dim, dtype=np.float64)
        outer_product = np.outer(x, x)

        J = identity - learning_rate * outer_product

        return J.astype(np.float64)

    #batch jacobian -- generalises jacobian_theta_update to a
    #multi-sample gradient step. For a single sample (n=1) this
    #reduces to exactly I - alpha * x x^T, matching the established
    #single-sample convention (and the paper's Eq. 15 chain term)
    #exactly. For n>1 it uses the mean-gradient Jacobian
    #I - (alpha/n) * X^T X, which is the correct parameter-recurrence
    #Jacobian for the combined-batch training step actually performed
    #once the rehearsal buffer participates in training (Algorithm 1
    #line 15: train on D_cache_{t-1} U D_reh_buf).
    def jacobian_batch_theta_update(self,
                                    X_batch: np.ndarray,
                                    learning_rate: float) -> np.ndarray:
        if X_batch.ndim != 2 or X_batch.shape[1] != self.feature_dim:
            raise ValueError("X_batch must be 2D with matching feature_dim")

        n = X_batch.shape[0]
        identity = np.eye(self.feature_dim, dtype=np.float64)
        gram = X_batch.T @ X_batch  # sum_i x_i x_i^T

        J = identity - (learning_rate / n) * gram

        return J.astype(np.float64)
    
    #clone
    def clone(self):
        new_model = LinearRegressionOnline(self.feature_dim)
        new_model.theta = self.theta.copy()
        return new_model

    def gradient_wrt_input(self, x: np.ndarray, y: float) -> np.ndarray:
        if x.shape[0] != self.feature_dim:
            raise ValueError("Input dimension mismatch")

        y_hat = np.dot(x, self.theta)
        error = y_hat - y

        grad_x = 2 * error * self.theta
        return grad_x.astype(np.float64)