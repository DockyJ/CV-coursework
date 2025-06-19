# CV-Project
主要包括基于 ResNet 的轻量级图像分类器开发与优化，
基于 Faster R-CNN 的多类口罩佩戴检测与计数


**Question 1 (14 marks)**

Write two functions _compute_gradient_magnitude(gr_im, kx, ky)_ and
_compute_gradient_direction(gr_im, kx, ky)_ to compute the magnitude and direction of
gradient of the grey image _gr_im_ with the horizontal kernel _kx_ and vertical kernel _ky_.

Inputs

- _gr_im_ is a 2-dimensional numpy array of data type _uint8_ with values in [0,255]
- _kx_ and _ky_ are 2 dimensional numpy arrays of data type _uint_.

Outputs

- The expected outputs are two 2-dimensional numpy array of the same shape as of
    _gr_im_ and of data type _float_.

Data

- You can work with the image at _data/shapes.png_.

Marking Criteria

- The outputs should exactly match with the correct gradient magnitude and gradient
    direction of a given grey image _gr_im_ to obtain the full marks. There is no partial
    marking for this question.

**Question 2 (20 marks)**

Write a function _generate_bovw_spatial_histogram(im, locations, clusters, division)_ to create
bag of visual words representation of an image _im_ whose features are located at locations and
the quantised labels of those features are stored in clusters. You have to build the histogram
based on the division information provided in division. For example, if division = [2, 3], you
have to imagine dividing the image along Y-axis in 2 parts and along X-axis in 3 parts (as
shown in the right most figure below), else if division = [2, 2], you have to imagine dividing
the image in 2 parts along both the axes, else if division = [1, 1], you just compute the bag-of-
visual-words histogram on the entire image without dividing into any parts. You have to
showcase three test cases to call the above function in order to calculate the bag-of-visual-
words spatial histograms on the image _im_ showcasing coarse and fine divisions.
![image](https://github.com/user-attachments/assets/a15d0927-baa3-4486-a83a-0665788db275)


Inputs

- _im_ is a 3-dimensional numpy array of data type _uint_.
- _locations_ is a 2-dimensional numpy array of shape 𝑁× 2	 of data type _int64_ , whose
    each row is a Cartesian coordinate (𝑥,𝑦).
- _clusters_ is a 1-dimensional numpy array of shape (𝑁,) of data type _int64_ , whose each
    element indicates the quantised cluster id.
- _division_ is a list of integers of length 2.

Outputs

- This function should return a 1-dimensional numpy array of data type _int64._ The
    results for all three test cases should be saved and submitted along with the
    information in the report on the division numbers and the corresponding output
    filenames.

Data

- There is no specific data for this question. However, you can create data using one of
    the images available inside the data folder.

Marking Criteria

- In each test case, your spatial histogram should be exactly matched with the correct
    spatial histogram to obtain the full marks. Coarser test cases contain lower weightage
    compared to their finer counter parts.

**Question 3 (10 marks)**

Write a function _compute_rotation_matrix(points, theta)_ to compute the rotation matrix in
homogeneous coordinate system to rotate a shape depicted with 2-
dimensional (𝑥,𝑦) coordinates points with an angle 𝜃 (theta in the definition) in the
anticlockwise direction about the centre of the shape.


Inputs

- _points_ is a 2-dimensional numpy array of data type _uint8_ with shape 𝑘× 2_._ Each row
    of points is a Cartesian coordinate (𝑥,y).
- _theta_ is a floating-point number denoting the angle of rotation in degree.

Outputs

- The expected output is a 2-dimensional numpy array of data type _float64_ with shape
    3 × 3

Data

- You can work with 2-dimensional numpy array at _data/points.npy_

Marking Criteria

- You will obtain the full mark if your rotation matrix exactly matches with the actual
    rotation matrix. If your matrix does not exactly match, you will not get any mark and
    there is no partial mark for this question.

**Question 4 (10 marks)**

Write a function _train_cnn(model, train_loader)_ to train the following version of the Residual
Network (ResNet) model on the EXCV10 (Exeter Computer Vision 10) dataset (available at
this link)

At the end of the training, this function should save the best weights of the trained CNN at:
_data/weights_resnet.pth_. The EXCV10 dataset contains 10000 images from 10 classes which
are further split into train (available at _train/_ folder; total 8000 images with 800 images/class)
and validation (available at _val/_ folder; total 2000 images with 200 images/class) sets. For
training your model, please feel free to decide your optimal hyperparameters, such as the
number of epochs, type of optimisers, learning rate scheduler etc within the function, which
can be done to optimise the performance of the model on the validation set.

Inputs

- _model_ is an instantiation of ResNet class which can be created as follows:
    _ResNet(block=BasicBlock, layers=[1, 1, 1], num_classes=num_classes)._ An example of
    this can be found in the below snippet.
- _train_loader_ is the training data loader. You can create the dataset and data loader for
    your training following the example in the snippet below. Feel free to try other data
    augmentation and regularisation techniques to train a better model.


Outputs

- This function should not necessarily return any output, instead it should save your best
    model at _data/weights_resnet.pth._

Data

- You can train your model on the data available
    at https://empslocal.ex.ac.uk/people/staff/ad735/ECMM426/EXCV10.zip. As EXCV
    dataset is quite large in size, do not upload it with your submission.

Marking Criteria

- You will obtain full marks if the model weights saved at _data/weights_resnet.pth_ can be
    loaded to a new instantiation of the model _ResNet(block=BasicBlock, layers=[1, 1, 1],_
    _num_classes=num_classes)_. You will not get any mark if your model is missing or
    saved in a different location or it cannot be loaded to the aforementioned model
    instance. Additionally, the quality of your trained model will be examined in the next
    question.

**Question 5 (15 marks)**

Write a function _test_cnn(model, test_loader)_ which will return the predicted labels by the
model that you trained in the previous question for all the images supplied in the _test_loader_
object. The test set will contain 3000 images (300 images/class) from the same distribution as
of the EXCV10 dataset.

Inputs

- _model_ is an instantiation of ResNet class which can created as follows
    _ResNet(block=BasicBlock, layers=[1, 1, 1], num_classes=num_classes)._
- _test_loader_ is the data loader containing test data. The test data loader can be created
    following the example in the cell below. We will only use vanilla transformation to the
    test dataset.

Outputs

- This function should return a 1-dimensional numpy array of data type _int_
    containing the predicted labels of the images in the _test_loader_ object.
- This function should also return the classification accuracy as a percentage


Data

- You can test your model on _val_ set of the data available
    at https://empslocal.ex.ac.uk/people/staff/ad735/ECMM426/EXCV10.zip. As EXCV
    dataset is quite large in size, do not upload it with your submission.

Marking Criteria

- Your model will be tested based on average classification accuracy on a test set of 3000
    images (300 images/class). You will obtain 50% marks if the obtained accuracy of your
    model on the test set is greater than or equal to 50%, 60% marks if your model obtains
    55% accuracy or more, 70% marks if your model gets 60% accuracy or more, 80%
    marks if your model acquires 65% accuracy or more, 90% marks if your model wins
    70% accuracy or more, and full marks if your model secures 75% accuracy or more.
    You will not obtain any mark if your model cannot achieve 50% accuracy.
- In the case you only provide the first output and not the classification accuracy, the
    abovementioned evaluation percentages will be calculated from 7.5 marks instead of 15
    marks.

**Question 6 (6 marks)**

Write _compute_confusion_matrix(ture,predictions)_ to compute the confusion matrix for the
output labels generated in Question 5.

Inputs

- _true_ is a 1-dimensional array of true labels for the test data in Question 5
- _predictions_ is a 1-dimensional array of outputs from Question 5

Data

- You can use the output and true labels of Question 5 as your data

Marking Criteria

- The output should exactly match the correct confusion matrix resulting from the
    output in Question 5 to obtain the full marks. There is no partial marking for this
    question.


**Question 7 (25 marks)**

Write a function _count_masks(dataset)_ which will count the number of faces correctly
wearing mask ( _with_mask_ class), without mask ( _without_mask_ class) and incorrectly wearing
mask ( _mask_weared_incorrect_ class) in the list of images dataset which is an instantiation of
the _MaskedFaceTestDataset_ class shown below. (Hint: You are expected to implement a 3
class (4 class with background) masked face detector which can detect the aforementioned
categories of objects in a given image. However, you are absolutely free to be more
innovative and come out with different solutions for this problem.)

![image](https://github.com/user-attachments/assets/82aad425-5dc5-46ee-9783-39792fdfa8b8)

Inputs

- _dataset_ is an object of the _MaskedFaceTestDataset_ class shown in the above code
    snippet.

Outputs

- This function should return a 2-dimensional numpy array of shape 𝑁× 3 of data type
    _int64_ whose values should respectively indicate the number of faces wearing mask,
    without mask and incorrectly wearing mask.


- This function should also return the Mean Absolute Percentage Error (MAPE) score
    using the below formula.

![image](https://github.com/user-attachments/assets/c53aca53-26ff-4e55-af0b-c7e74876ba9a)

Data

- You can train and test your model on the data available at
    https://empslocal.ex.ac.uk/people/staff/ad735/ECMM426/MaskedFace.zip. This dataset
    contains some images and corresponding annotations (locations together with category
    information) of masked faces, which are split into _train_ and _val_ subsets. You can train
    your model on _train_ set and decide your hyperparameters on the _val_ sets. As
    MaskedFace dataset is quite large in size, please do not upload it with your submission.

Marking Criteria

- You will obtain 50% marks if the obtained average MAPE of your model on the test
    set is lower than or equal to 30%, 62.5% marks if your model obtains 25% MAPE or
    less, 75% marks if your model gets 20% MAPE or less, 87.5% marks if your model
    acquires 15% MAPE or less, and full marks if your model secures 10% MAPE or
    less. You will not obtain any mark if your model cannot achieve 30% MAPE.
- In the case, you only provide the first output and not the MAPE score, your
    abovementioned evaluation percentages will be calculated from 12.5 marks instead of
    25 marks.


