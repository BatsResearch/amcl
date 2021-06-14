# Adversarial Multi Class Labeling (AMCL)

Lightweight implementations of an adversarial learning scheme for multi class weakly supervised learning tasks.

# Usage

To use our methods, you first need to import the subgradient_method file with 


```
import algorithms.subgradient_method as SG
```

Next, you should generate the constraints for the linear program. You can do this by calling

```
constraint_matrix, constraint_vector, constraint_sign = SG.compute_constraints_with_loss(loss_function, 
																						 unlabeled_data, 
																						 labeled_data, 
																						 labels)

```

Then, you can run the linear program as follows:

```
initial_theta = np.random.normal(0, 0.1, (np.shape(test_data)[1], C))
model_theta = SG.subGradientMethod(unlabeled_data, constraint_matrix, constraint_vector, 
								   constraint_sign, loss_function, model, 
								   projection_function, initial_params, 
								   T, h, N, num_unlab, C)
```

Here, this allows for different parameterized models by specifying the model and initial_params values. You can also change the objective funtion of the linear program
by changing the particular loss_function parameter. Finally, there are a few other parameters that specify the training procedure, which you can cusotmize (T, projection_function, etc.). 


You can see an example usage of our methods on a toy dataset in the main.py file. 