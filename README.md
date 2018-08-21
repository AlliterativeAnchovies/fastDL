# fastDL
My small library to quickly grab and prepare data for machine learning, built on top of fastAI

If you want to see how to use it, check out fastDL_examples.ipynb - I would recommend opening it in Colab because the github view doesn't format well.

The general workflow will look like this:
1) Download the dataset you want to train a model on
2) Create an instance of the ModelDirectory class
3) Declare a "Task" for your ModelDirectory
4) Tell the Task to create and train a fastai model.

Right now the library is fairly limited - it only works for images and text data (especially images) and there isn't much customizability.  However, if you need to customize it more I made sure the Task class has every internal variable exposed and relatively clearly named, so you can manually edit things between the steps.

I am currently updating this fairly frequently.  My current goal is have the library allow you to easily predict image data and explore the false positives/etc, after that I'll focus more on text.  Ultimately the goal is to have everything that is done in the fastAI course be easily doable here in minimal code.  However, this isn't meant to replace fastAI, this is merely a wrapper around it to help with data loading and visualization.

Dependencies: fastAI, spacy

Here is a quick example of code you can run in colaboratoy right now to make a dogs/cats classifier like is done in the first fastAI lesson:

----

!wget http://files.fast.ai/data/dogscats.zip

modelDirec = ModelDirectory("dogsVcats",trainingDataPath = "dogscats.zip",validationDataPath = "")

modelDirec.extractJumbledValidation(["dogscats","train"],"dogscats/valid/.")

theTask = modelDirec.giveTask("classification","dogscats/train",folderCategories=["cats","dogs"])

theTask.prepImageDataForFastAI()

theTask.createModel()

theTask.trainModel()

#Model is now fully trained.  You can directly access it with "theTask.fastAIModel"

-----

I apologize if the fastDL_examples notebook becomes out of date, I am updating this very frequently.  I will try to keep it up to date, but if something every goes wrong, feel free to check out the source code to see if a method's parameters have changed.
