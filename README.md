# End-to-end Segmentation of Brain Tumor and Deployment

In this work, I develop a segmentation network using U-Net to segment the tumor from the input image. This pipeline is hosted on Heroku with continuous integration.

I use the dataset from Kaggle: https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

I have written the UNET code with inspirations from various places on the internet.

I trained this network on my Laptop for 10 epochs using AdamW Optimizer with learning rate of 1e-4.

### Results
![result](https://user-images.githubusercontent.com/22027039/116832322-f783b900-abb4-11eb-8361-4f19cce14f8b.PNG)

As we can see from the above results, within just 10 epochs, the model starts to generalize well.
