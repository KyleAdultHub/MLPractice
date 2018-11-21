from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model

# VGG16模型,权重由ImageNet训练而来,不保留顶层的3个全连接网络
base_model = VGG16(weights='imagenet', include_top=False)

# 自定义一个简单的分类器
x = base_model.output
x = x.add(GlobalAveragePooling2D())
x = x.add(Dense(2048, activation='relu'))
x = x.add(Dropout(0.5))
predictions = Dense(1000, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 预训练模型部分,权重设置为不训练
for layer in base_model.layers:
    layer.trainable = False

# 编译
model.compile(optimizer='sgd', loss='categorical_crossentropy')

# 传入图像，并进行预处理
img_path = './wolf.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 预测
preds = model.predict(x)
print(preds)
print('Predicted:', decode_predictions(preds, top=5)[0])

