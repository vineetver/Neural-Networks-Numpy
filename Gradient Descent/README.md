## Mini-Batch

Here is the code for a function that implements Mini-batch gradient descent.

```python
def mini_batch_gradient_descent(self, X, Y):
        """
        This function performs mini-batch gradient descent

        Parameters
        ----------
        X : 4d-array
            Training data
        Y : 1d-array
            Training labels
        """    
        seed(19)                               
        m = len(X)  
        self.theta_ = np.random.randn(X.shape[1])                                    ## number of training examples
        start_time = time.time()
        self.eta_ = self.learning_rate 

        for epoch in range(self.epochs):    
            error = 0                                                
            random_mini_batch = np.random.permutation(m)           
            X = X[random_mini_batch]                               
            Y = Y[random_mini_batch]
            for i in range(0, m, self.mini_batch_size):     
                xi = X[i : i + self.mini_batch_size]            
                yi = Y[i : i + self.mini_batch_size]            
                nabla_mse = 2/m * xi.T.dot(xi.dot(self.theta_) - yi)        ## Calculate the nabla_mse   xi^T . (xi . theta - yi) using the random mini-batches of X and Y
                self.theta_ -= self.eta_ * nabla_mse                        ## Update theta using the nabla_mse  (eta * nabla_mse)
                error = self.calculate_loss(Y, self.predict(X))              ## Calculate the MSE 
            self.errors_.append(error)
            print(
                f'Epoch: {epoch + 1} [==========================] Time: {time.time() - start_time:.2}s | Error: {error * 100:.2f}')
        return self
```

## Batch 

Here is the code for a function that implements Batch gradient descent.

```python
def batch_gradient_descent(self, X, Y):
        """
        This function performs batch gradient descent
        """
        self.theta_ = np.random.randn(X.shape[1])
        self.eta_ = self.learning_rate
        m = len(X)
        start_time = time.time()

        for epoch in range(self.epochs):
            nabla_mse = 2/m * X.T.dot(X.dot(self.theta_) - Y)  ## Calculate the nabla_mse   X^T . (X . theta - Y) using X and Y (whole training data)
            self.theta_ -= self.eta_ * nabla_mse               ## Update theta using the nabla_mse  (eta * nabla_mse)
            error = self.calculate_loss(Y, self.predict(X))      ## Calculate the error
            self.errors_.append(error) 
            print(
                f'Epoch: {epoch + 1} [==========================] Time: {time.time() - start_time:.2}s | Error: {error * 100:.2f}') 
        return self
```

## Stochastic 

Here is the code for a function that implements Stochastic gradient descent.

```python
def stochastic_gradient_descent(self, X, Y):
        """
        Perform stochastic gradient descent
        """
        seed(42)
        self.theta_ = np.random.randn(X.shape[1])
        self.eta_ = self.learning_rate
        m = len(X)
        start_time = time.time()

        for epoch in range(self.epochs):
            error = 0
            for i in range(m):
                random_index = i
                xi = X[random_index:random_index+1]     ## get one random value of X 
                yi = Y[random_index:random_index+1]     ## get one random value of Y
                nabla_mse = 2/m * xi.T.dot(xi.dot(self.theta_) - yi)    ## Calculate the nabla_mse   xi^T . (xi . theta - yi) using one random value of X and Y 
                self.theta_ -= self.eta_ * nabla_mse                ## Update theta using the nabla_mse  (eta * nabla_mse)
                error = self.calculate_loss(Y, self.predict(X))      
            self.errors_.append(error)
            print(
                f'Epoch: {epoch + 1} [==========================] Time: {time.time() - start_time:.2}s | Error: {error * 100:.2f}')
        return self
```