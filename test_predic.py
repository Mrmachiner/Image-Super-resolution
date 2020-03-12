
import numpy as np
from os.path import expanduser
from PIL import Image
from ISR.models import RDN, RRDN

# RDN: psnr-large, psnr-small, noise-cancel
# RRDN: gans

model = RDN(weights='psnr-large')

img = Image.open(expanduser('/home/minhhoang/image-super-resolution/data/input/sample/test.jpg'))
lr_img = np.array(img)

sr_img = model.predict(lr_img,)
output = Image.fromarray(sr_img)
output.save(expanduser('data/test_psnr-large.png'))