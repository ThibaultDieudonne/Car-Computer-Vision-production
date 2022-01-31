from flask import Flask, send_file, request
import os
import io
import imageio
from PIL import Image
import numpy as np
import tensorflow as tf
import segmentation_models as sm


# LOAD MODEL

size = (224, 224)

BACKBONE = 'resnet34'

preprocess_input = sm.get_preprocessing(BACKBONE)

AUTOTUNE = tf.data.AUTOTUNE

BUFFER_SIZE = 1000

def load_model(name="unet_resnet34_aug"):
    model = sm.Unet(BACKBONE, classes=8, input_shape=(None, None, 3), activation='softmax', encoder_weights=None)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    return model

api_model = load_model()
api_model.load_weights("./models/unet_resnet34_aug_weights.h5")


# LOAD DATA

imgs_path = "./data/imgs"

lbls_path = "./data/lbls"

img_path = []

lbl_path = []

for _, _, files in os.walk(imgs_path):
    for name in files:
        if name.endswith(("_leftImg8bit.png")):
            img_path += ["/".join([imgs_path, name])]
            lbl_path += [f"./data/lbls/{name}".replace("leftImg8bit", "gtFine_labelIds")]

color_map = np.array([[0, 0, 0], # black: void
          [153, 153, 0], # yellow: flat
          [255, 204, 204], # light pink: construction
          [255, 0, 127], # strong pink: object
          [0, 255, 0], # green: nature
          [0, 204, 204], # light blue: sky
          [255, 0, 0], # red: human
          [0, 0, 255], # blue: vehicle
         ])

def load_imgs(path1, path2):
    img1 = tf.io.read_file(path1)
    img1 = tf.io.decode_png(img1)
    img2 = tf.io.read_file(path2)
    img2 = tf.io.decode_png(img2)
    
    img1 = preprocess_input(img1)
    img1 = tf.cast(img1, tf.float32)
    img2 = tf.squeeze(img2, axis=-1)
    img2 = tf.one_hot(img2, 8, axis=-1)
    
    return img1, img2

def get_imgs(loc=0):
    pred_ds = tf.data.Dataset.from_tensor_slices(
        ([img_path[loc]],
         [lbl_path[loc]],)
        ).shuffle(BUFFER_SIZE)
    pred_ds = pred_ds.map(load_imgs, num_parallel_calls=AUTOTUNE)
    pred_ds = pred_ds.batch(1)
    pred = api_model.predict(pred_ds)
    toimg = np.asarray(pred[0])
    img3 = np.argmax(toimg, axis=2)
    img1 = imageio.imread(img_path[loc])
    img2 = imageio.imread(lbl_path[loc])
    img2 = color_map[img2]
    img3 = color_map[img3]

    return np.vstack((img1, img2, img3))


# SERVE API

app = Flask(__name__)

def is_valid(arg):
    if isinstance(arg, str):
        if arg.isdigit():
            if int(arg) >= 0 and int(arg) < len(img_path):
                return 1
    return 0

@app.route("/")
def hello():
    img_id = request.args.get('id')
    if is_valid(img_id):
        img_id = int(img_id)
        img = get_imgs(img_id)
        img = Image.fromarray(img.astype('uint8'))
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')  
        file_object.seek(0)

        return send_file(file_object, mimetype='image/PNG')
    
    else:
        
        return f"Provide an id in URL (?id=X)"