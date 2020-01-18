#!/usr/bin/env python3
from PIL import Image
from StyleTransfer import StyleTransfer
import numpy as np
from time import time

style = Image.open("data/style/2.jpg")
style = np.asarray(style.resize((576, 1024))).transpose(2,0,1)
s = StyleTransfer()
s.set_style(style, 1.0)
tt = time()
im = Image.open("data/content/057725.jpg")
im = np.asarray(im.resize((576, 1024))).transpose(2,0,1)
for _ in range(100):
    t = s.stylize_frame(im).transpose(1,2,0)
print(100/(time() - tt))
print(t.shape)
Image.fromarray(t).show()

