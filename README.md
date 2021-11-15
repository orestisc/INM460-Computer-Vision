# Face Emotion Recognition
This project uses Pytorch Framework for human face emotions detection. The project can run on images from the test set as well on the any video or live camera. 
The code first extracts the face from an image and then feed the face into the model for classification. 
The code implements model building and training, model evaluation on test set and loading data using data loaders.

#################
###File Order ###
CV_CW-->
	CW_Datset.zip --> # To be unzipped in colab for SVM and MLP training.
                	train -->
                    		image_1.jpg, image_2.jpg, .... n
                	test -->
                    		image_1.jpg, image_2.jpg, .... n
                	labels --> 
                    		labels_txt_train.txt
                    		labels_txt_test.txt
	Project-->
		configs -->
			classes.txt # contains the names of classes based on line number e.g. class 0 will be mapped with the name at the first line in classes.txt and so on.
		Models --> # Contains all machine leanrning algorithms trained with their hyperparameters.
			SVM_HOG
			SVM_SIFT
			CNN
			MLP_HOG
		Results --> # Screenshots of models training
			CNN
			SVM_grid
			MLP_grid
		Video --> 
			video_elon.mp4
		Config.py	# The following files are used for the training and validation of CNN model. CNN model was trained locally
		dataset.py
		EmotionRecognitionVideo.py
		model.py
		train.py
		scikit_learn_training.ipynb # Containing SVM and MLP algorithms and training results.
		test_functions.ipynb # Contains functions for running and visualising all trained models.
		

####Note####
Run test_functions.ipynb to test results.
			
## CONFIG.py
This file is to specify parameters which are being used in the entire project. 
You must alter those parameters according to your dataset and other paths. For Example, in training parameters, you'd need to modify the number of classes, number of epochs, batch size and so on.

If RESUME is true, the code will first try to load the checkpoint specified as CHECKPOINT_PATH. All checkpoints during training will be saved in CHECKPOINT_DIR.

