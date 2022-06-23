## Perceptron learning rule

Here is the code for a function that implements perceptron learning rule.

```python
def fit(self, X, Y):
        """
        This function trains the perceptron.
        """
        self.b_ = 0
        self.X = X
        self.Y = Y
        start_time = time.time()
     
        for epoch in range(self.epochs):
            """
            We can also use the Normal Equation to solve this problem directly.
            """                   
            error = 0
            for x, y in zip(X, Y):
                nabla_mse_perceptron = self.learning_rate * (y - self.predict(x))  
                self.theta_ += nabla_mse_perceptron * x                               
                self.b_ += nabla_mse_perceptron       

            error = self.calculate_loss(y, self.predict(x))             
            self.errors_.append(error)
            print(f'Epoch: {epoch + 1} [==========================] Time: {time.time() - start_time:.2}s | Error: {error * 100:.2f}')

        return self
```
