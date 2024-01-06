from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from os import getcwd
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.xception import Xception, preprocess_input
import numpy as np
import os
import tensorflow as tf
 
with tf.device('/GPU:0'):
    # Your TensorFlow operations here

     print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Set the variables
image_size = [299, 299]
trainpath = r'E:\dataset\training'
testpath = r'E:\dataset\testing'

# Create ImageDataGenerators for both training and testing sets with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)



test_datagen = ImageDataGenerator(rescale=1./255)

if not os.path.exists(trainpath):
    print(f"Directory not found: {trainpath}")
else:
    # Flow from directory for training set
    training_set = train_datagen.flow_from_directory(
        trainpath,
        target_size=(image_size[0], image_size[1]),
        batch_size=32,
        class_mode='categorical'
    )
print("Hiiiii")
# Flow from directory for testing set
test_set = test_datagen.flow_from_directory(
    testpath,
    target_size=(image_size[0], image_size[1]),
    batch_size=32,
    class_mode='categorical'
)

xception=Xception(input_shape=(image_size[0], image_size[1],3) ,weights='imagenet',include_top=False)
for layer in xception.layers:
  layer.trainable=False
  x=Flatten()(xception.output)


prediction =Dense(5, activation='softmax')(x)

model=Model(inputs=xception.input,outputs=prediction)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

model.save('updated_xception_diabetic_retinopathy.h5')
model.summary()
app = Flask(__name__)

# Load the trained model


@app.route('/')
def home():
    print("Current working directory:", getcwd())
    print("Templates:", app.jinja_loader.list_templates())
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['file']

    # Preprocess the image
    img = image.load_img(image_file, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make a prediction using the loaded model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Return the prediction result as JSON
    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
