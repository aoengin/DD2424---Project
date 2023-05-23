# DD2424-Project

This is the final version of the code used for the project assignment of course DD2424 given at KTH Royal Institute of Technology. 
# Setup
For this program to run correctly, the uncompressed [Oxford IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) must be present in the folder named *oxford-iiit-pet*. Then the command
> python generate_datasets.py
must be run to generate small, medium and large sized (with respect to the amount of training data) datasets. (Beware! This will take up ~5 GBs of space in your hard disk)

# Executing the Program
The program can simply be run with the following command:
> python image_classification.py

# Command-line arguments
Additionally, the following arguments can be given to the program for various effects:
-t (small|medium|large) := Choose the size of the training dataset (Default large)
-b := Retrain batch normalization layers (Default True)
-d := Use data augmentation (Default True)
-n (1|2|3|4|5) := The number of stages retrained (Default 2)
-c (2|37) := The number of classes in the dataset (Default 2)
-e (number) := The number of epochs to train the model for (Default 15)
--sophisticated_data_augs (none|cutmix|mixup|erase) := Extra option for more sophisticated data augmentations (Default none)
--only_update_bn_params := Option to train by updating only the batch normalization parameters (Default False)
