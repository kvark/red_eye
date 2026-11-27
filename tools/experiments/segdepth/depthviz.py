import sys, numpy as np
from PIL import Image
d = np.load(sys.argv[1])
key = sys.argv[2]; out = sys.argv[3]
x = d[key].reshape(d[key].shape[-2:]) if d[key].ndim > 2 else d[key]
H, W = x.shape
img = Image.open('ref/dog.jpg').convert('RGB').resize((W, H))
n = (x - x.min()) / (x.max() - x.min() + 1e-9)
v = (n * 255).astype(np.uint8)
rgb = np.stack([v, (v * 0.3).astype(np.uint8), np.zeros_like(v)], -1)
Image.fromarray(np.concatenate([np.asarray(img), rgb], 1)).save(out)
print('shape', x.shape, 'min %.3f max %.3f' % (x.min(), x.max()))
def reg(name, y0, y1, x0, x1):
    r = n[int(y0*H):int(y1*H), int(x0*W):int(x1*W)]
    print(f'  {name:28s} mean normalised {r.mean():.2f}')
# dog.jpg 768x576: dog ~ x 0.17-0.42, y 0.38-0.95; bicycle x 0.16-0.75 y 0.23-0.75; truck x 0.6-0.9 y 0.13-0.3
reg('dog (x.18-.40,y.55-.90)', .55, .90, .18, .40)
reg('porch floor bottom (y.93-1)', .93, 1.0, .0, 1.0)
reg('bicycle frame (x.35-.6,y.35-.6)', .35, .6, .35, .6)
reg('truck (x.62-.88,y.14-.28)', .14, .28, .62, .88)
reg('top-left wall/street (y0-.1,x0-.3)', 0, .1, 0, .3)
