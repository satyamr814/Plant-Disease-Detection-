## README.md

# Plant Disease Detection

A convolutional neural network that identifies plant diseases from photos of leaves. Built with TensorFlow and trained on the PlantVillage dataset.

## What it does

Take a photo of a plant leaf. The model tries to tell you what's wrong with it, if anything.

It recognizes 38 different plant-disease combinations - mostly common crops like tomatoes, peppers, potatoes, grapes, and cherries. The model trains on 234x234 pixel images, looking for patterns in color and texture that indicate specific diseases.

## Dataset

The [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) on Kaggle has something like 54,000 images of healthy and diseased leaves. Powdery mildew, early blight, bacterial spots, various molds - the usual suspects for these crops.

## Quick Start

### Requirements

```bash
pip install tensorflow pillow numpy matplotlib imagesize
```

You'll also need a Kaggle account to download the dataset. Create a `kaggle.json` file with your API credentials.

### Running it

Clone the repo, add your `kaggle.json` file, and open the notebook in Jupyter or Colab. Run the cells top to bottom. It downloads the dataset, splits it 80/20 for training and validation, builds a CNN, trains for 5 epochs, and evaluates the results.

## Model Architecture

Pretty straightforward CNN. Starts with a Conv2D layer (32 filters, 3x3), adds MaxPooling, stacks a few more conv and pooling layers, then feeds into dense layers for the final classification. Adam optimizer, categorical crossentropy loss.

## Results

Validation accuracy bounces around but usually ends up somewhere between 85-95% after 5 epochs. Not bad for a basic CNN. You could probably push it higher with more training time or some data augmentation tricks.

## Usage

After training, you can predict on new images:

```python
from PIL import Image
import numpy as np

# Load and preprocess
img = Image.open('path/to/leaf.jpg')
img = img.resize((234, 234))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
disease_name = class_indices[predicted_class]

print(f"Detected: {disease_name}")
```

## Limitations

Only recognizes plants that were in the training set. Needs decent photos - clear lighting, single leaf in frame. Won't spot early infections before visual symptoms show up. If your tomato plant looks weird but doesn't match any training examples, the model just guesses.

## Credits

Dataset: PlantVillage (CC BY-NC-SA 4.0)  
Built with TensorFlow/Keras

---

## License

This project is for educational purposes. The PlantVillage dataset has its own license (CC BY-NC-SA 4.0).
