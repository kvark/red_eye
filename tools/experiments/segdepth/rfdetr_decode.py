import sys, json, numpy as np
from PIL import Image
d = sys.argv[1]
z = np.load(f'{d}/outputs.npz')
boxes, logits, masks = z['pred_masks'][0], z['pred_boxes'][0], z['pred_logits'][0]
cfg = json.load(open('ref/segdepth/dl/rfdetr_config.json'))['id2label']
prob = 1 / (1 + np.exp(-logits))
img = Image.open('ref/dog.jpg').convert('RGB'); W, H = img.size
over = np.asarray(img).astype(np.float32).copy()
q, c = np.unravel_index(np.argsort(-prob, axis=None)[:300], prob.shape)
seen = set(); rows = []
colors = [(255,0,0),(0,255,0),(0,128,255),(255,255,0),(255,0,255),(0,255,255)]
for qi, ci in zip(q, c):
    if prob[qi, ci] < 0.3 or qi in seen: continue
    seen.add(qi)
    cx, cy, w, h = boxes[qi]
    m = masks[qi] > 0
    mi = np.asarray(Image.fromarray((masks[qi]).astype(np.float32)).resize((W, H), Image.BILINEAR)) > 0
    ys, xs = np.nonzero(mi)
    mb = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
    box = (int((cx-w/2)*W), int((cy-h/2)*H), int((cx+w/2)*W), int((cy+h/2)*H))
    rows.append((cfg[str(ci)], round(float(prob[qi, ci]), 3), 'box', box, 'mask px', int(mi.sum()), 'mask bbox', mb, 'mask/box area %.2f' % (mi.sum() / max(1, (box[2]-box[0])*(box[3]-box[1])))))
    col = np.array(colors[len(rows) % len(colors)], np.float32)
    over[mi] = over[mi] * 0.5 + col * 0.5
for r in rows: print(*r)
Image.fromarray(over.astype(np.uint8)).save(f'{d}/masks.png')
