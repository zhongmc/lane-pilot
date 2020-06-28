import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

num_skipped = 0
for folder_name in ('free', 'blocked'):
    folder_path = os.path.join('dataset', folder_name )
    for fname in os.listdir(folder_path ):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, 'rb')
            is_jfif  = tf.compat.as_bytes('JFIF') in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            num_skipped += 1
            #os.remove( fpath )
    print("%d images is not jfif !" % num_skipped )

    image_size = (244, 244) 
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset',
        validation_split = 0.2,
        subset="training",
        seed = 1337,
        image_size = image_size,
        batch_size = batch_size,

    )

    val_ds =  tf.keras.preprocessing.image_dataset_from_directory(
        'dataset',
        validation_split = 0.2,
        subset="validation",
        seed = 1337,
        image_size = image_size,
        batch_size = batch_size,
    )
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(int(labels[i]))
        plt.axis('off')

data_argumentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RadomRotation(0.1),

    ]
)

plt.figure(figsize=(10,10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_argumentation( images)
        ax = plt.subplot(3,3, i+1)
        plt.imshow(augmented_images[0].numpy.astype("uint8"))
        plt.axis("off")

#preprocess the data
# inputs = keras.Input(shape = input_shape )
# x = data_augmentation(inputs)
# x = layers.experimental.preprocessing.Rescaling(1./255)(x)

#config dataset for performance
train_ds = train_ds.prefetch(buffer_size = 32)
val_ds = val_ds.prefetch(buffer_size = 32 )

#build a model 
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape = input_shape)
    x = data_augmentation(inputs)
    #Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0/255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding = "same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    previous_block_activation = x  #set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.maxPooling2D(3, strides= 2, padding="same")(x)

        #project residual
        residual = layers.Conv2D(size, 1, strides = 2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual]) #add back residual
        previous_block_activation = x #ser aside nex residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense( units, activation=activation)(x )
    return keras.Model( inputs, outputs)

model = make_model( input_shape = image_size + (3, ), num_classes = 2)
keras.utils.plot_model(model, show_shapes = True)

#train the model
epochs = 50
callbacks =[
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss = "binaery_crossentropy",
    metrics = ["accuracy" ],
)

model.fit( 
    train_ds, epochs = epochs, callbacks = callbacks, validation_data=val_ds,
)


model.save('collision.h5')

img = keras.preprocessing.image.load_img(
    'dataset/blocked/06-14-163459.jpg', target_size = imge_size
)

img_array = keras.preprocessing.image.img_to_array(img )
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict( img_array )
score = predictions[0]

print(
    "this image is %.2f percent blocked and %.2f percent free." % (100 * (1-score), 100 * score )
)