## **SVM Programming Report**
In this assignment we implemant various settings of support vector machine, including the regular linear SVM, SVM with slack variable, SVM with kernels.

## **SVM Formulation**
The objective function of support vector machine is:

$$
min_{w, b} \frac{1}{2}||w||^{2} \\
s.t. y_{i}(w^{T}x_{i} + b) \ge 1 \forall i
$$

## **Regular SVM**
So in our project we use the "w" to denote the weight, "b" to denote the bias, and x_{i} to denote training data, y_{i} to denote the training label.

So in regular SVM we set the slack variable to be a relatively very small number, and set the kernel to be linear.
```python
clf = svm.SVC(C=1e5, kernel = 'linear')
```
then we can get the following output to notice that class 0  "Iris-setosa" is linear separable
![](https://i.bmp.ovh/imgs/2022/04/03/1c378e8a14749b5f.png)

By training the regular SVM by the setting mentioned above we can get the following result:
```
training_error:0.016667
testing_error:0.000000
w_of_setosa:-0.046258538085542256, 0.5211827995531007, -1.0030446153124941, -0.4641297849669344, 
b_of_setosa: 1.452844
support_vector_indices_of_setosa: 13, 31, 34, 
w_of_versicolor: -0.15228426789115657, 0.13536379368102813, -0.490693752093727, -0.23688663894179923, 
b_of_versicolor: 2.289340
support_vector_indices_of_versicolor: 50, 52, 57, 63, 78, 
w_of_virginica: 4.264086882816628, 6.192174636991695, -8.641446592053398, -12.560510518902447, 
b_of_virginica: 19.189122
support_vector_indices_of_virginica: 97, 99, 103, 108, 
```

## **SVM with Slack Formulation**

A SVM with slack variable look like this, we allow little portion of data not in the decision region.
![](https://i.bmp.ovh/imgs/2022/04/03/8f90471dd8899712.png)

So the optimization problem formulation look like this:
![](https://i.bmp.ovh/imgs/2022/04/03/6c86c08324f36146.png)
we can then derive its dual lagurangian and dual problem 
![](https://i.bmp.ovh/imgs/2022/04/03/0988594784420a24.png)

the solution $\alpha_{i}$ has the following 3 cases:
1. $\alpha_{i} = 0$ the corresponding data are correctly classified and doesn’t contribute to the classifier, locating outside of the margin
2. 0 < αi < C: in this case, µi > 0 due to αi = C − µi
; Since µiξi = 0,
then we have ξi = 0. The corresponding data are correctly classified and
contributes to the classifier, locating on the margin
3. αi = C: in this case, µi = 0; then we have ξi > 0. The corresponding data
contributes to the classifier, locating inside the margin
If ξi ≤ 1, then the data is still correctly classified, not crossing decision
boundary
If ξi > 1, then the data is incorrectly classified, crossing decision boundary

And we can derive the bias in the following way:
![](https://i.bmp.ovh/imgs/2022/04/03/9a1e384c47d43494.png)

so in this project we tested varios slack varible from 0 to 1 step size 0.1

we get the following result:
we select slack variable 0.1 as example
```
training_error:0.016667
testing_error:0.033333
w_of_setosa:-0.08461491940959265, 0.4452659076269341, -0.8404228999803351, -0.39549567986979894, 
b_of_setosa: 1.567953
support_vector_indices_of_setosa: 13, 14, 31, 34, 
w_of_versicolor: -0.15228426789115657, 0.13536379368102813, -0.490693752093727, -0.23688663894179923, 
b_of_versicolor: 2.289340
support_vector_indices_of_versicolor: 43, 46, 48, 50, 52, 53, 56, 57, 58, 63, 64, 65, 66, 67, 71, 73, 78, 
w_of_virginica: 0.12864428210513168, 0.42231488821646934, -1.5263467916532356, -1.346810447700804, 
b_of_virginica: 7.685070
support_vector_indices_of_virginica: 80, 81, 83, 89, 91, 93, 96, 97, 103, 104, 108, 111, 112, 116, 117, 119, 

```

## **SVM with non-linear kernel**
So in this part we set the slack variable to be 1, then we try to use kernels including: 2nd-order polynomial kernel, 3rd-order polynomial, rbf, sigmoid.

We use non-linear kernel because SVM can only handle linearly separable data, so we can add non-linearity to handle higher order polynomial data.

So we can formulate the SVM with kernel in the following manner:
![](https://i.bmp.ovh/imgs/2022/04/03/06840ff7b5dae46b.png)

in the end we get the following result, take sigmoid as example:
```
training_error:0.958333
testing_error:0.966667
b_of_setosa: 0.328565
support_vector_indices_of_setosa: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
b_of_versicolor: 0.214481
support_vector_indices_of_versicolor: 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 
b_of_virginica: 0.014337
support_vector_indices_of_virginica: 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 

```
