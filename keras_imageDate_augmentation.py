from  keras.preprocessing import image

import matplotlib.pyplot as plt
import glob
from PIL import Image

import keras

print(keras.__version__)
datagen=image.ImageDataGenerator(rotation_range=30)


gen_data=datagen.flow_from_directory('./imgdata/',
                                      batch_size=1,
                                      shuffle=False,
                                       save_to_dir='./imgdata/pp/',
                                       save_prefix='gen',
                                       target_size=[224,224])

for i in range(9):
   gen_data.next()

name_list=glob.glob('./imgdata/pp/*')
fig=plt.figure()
for i in range(9):
    img=Image.open(name_list[i])
    sub_img=fig.add_subplot(331+i)
    sub_img.imshow(img)
plt.show()



