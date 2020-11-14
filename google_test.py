#from SYS6016.flask_hw4.flask_assignment4.google_pred import predict_plant
from PIL import Image
import numpy as np
from google_pred import predict_plant
#import os

# credential_path = 'velvety-transit-295121-2db38d88f2f7.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path


img = Image.open('0000002780.jpg')
img = img.resize((224,224))
img_array = np.array(img)
print(img_array.shape)
img_array = np.expand_dims(img_array,0)
print(img_array.shape)
print('Predicted Plant: ' + str(predict_plant(img_array)))
		
