# Housepricing
Keggle house pricing competition using RBFs

The test and training dataset is mapped to a [0,1] hypercube and a RBF function is created using the training set. Then the test set is estimated by simply mapping it to the hypercube, inserting the values in the RBF function and then map then back to the original values. Some different RBF kernels were used, including linear, cubic, thin plate, inverse and default multiquadratic. Multiquadratic performed best with epsilon = 0, and datacolumns with correlation < 4% and columns removed with > 10% data missing. 
