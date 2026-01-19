# CNN Based Mel-Spectrogram Analysis and Speaker Recognition
A project for Biometric Systems in the second year (2025/2026), first semester, of the master's degree in Computer Science at La Sapienza university of Rome.
The goal of the project is to implement a biometric system capable of executing an identification task in an open set scenario, through the use of mel-spectrograms and a CNN using the CosFace loss.

# Structure

- Inside Preprocessing.py there are the functions used in the preprocessing part of the pipeline
- SplitTestData.py has a function to decide how to split the data for the gallery and the probe set
- In model.py the entire method is executed, from creating datasets to creating and training the model
- In test.py all tests and metrics are created and executed
- gallery.json is an example of how the gallery looks like in json format
