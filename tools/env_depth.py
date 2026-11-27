#!/usr/bin/env python3
"""Check how the Quest stores its environment depth, from the app's dump.

With `/data/local/tmp/red_eye/dump_depth` present, the app logs its first
environment depth frame (a 40x40 subsample per eye) as `envdump` lines:

    adb shell touch /data/local/tmp/red_eye/dump_depth
    adb logcat -d -v raw -s red_eye:I | grep envdump > envdump.txt
    python tools/env_depth.py envdump.txt

Both eyes' images see the same surfaces from views 6 cm apart, so under
the right decoding of the depth, points from one land where the other sees
them. The script reprojects every left-eye point into the right eye under
each candidate encoding (OpenGL, Vulkan or reversed depth) and reports how
well the eyes agree, relative to distance, with each decoding's median
distance. A decoding can agree with itself and still be wrong: reversed
depth makes the room a shell a hand's width away, which only its median
gives away. The app assumes OpenGL depth (`perception::environment`).

Stereo cannot tell which way up the image is: the eyes sit side by side,
so flipping both vertically keeps them consistent. The app judges that
against the camera's own depth instead, and logs it ("agrees with the
camera").
"""

import re
import sys

import numpy as np


def quat_rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def parse(path):
    meta, views, rows = {}, [], {}
    floats = r"\[([^\]]*)\]"
    for line in open(path, encoding="utf-8"):
        if "envdump" not in line:
            continue
        line = line[line.index("envdump") :]
        if line.startswith("envdump size"):
            m = re.match(r"envdump size (\d+) (\d+) near (\S+) far (\S+)", line)
            meta = dict(
                w=int(m[1]), h=int(m[2]), near=float(m[3]), far=float(m[4].replace("inf", "inf"))
            )
        elif line.startswith("envdump view"):
            q, p, fov = [np.array([float(v) for v in g.split(",")]) for g in re.findall(floats, line)]
            views.append((q, p, fov))
        elif line.startswith("envdump row"):
            _, _, layer, y, hexes = line.split()
            rows[(int(layer), int(y))] = [int(hexes[i : i + 4], 16) / 65535.0 for i in range(0, len(hexes), 4)]
    w, h = meta["w"], meta["h"]
    raw = np.array([[rows[(l, y)] for y in range(h)] for l in range(2)])
    return meta, views, raw


ENCODINGS = ("OpenGL", "Vulkan", "reversed")


def linear(raw, near, far, encoding):
    """Metres ahead for stored values under each candidate encoding. With
    an infinite far plane, OpenGL and Vulkan depth decode alike."""
    d = np.where((raw > 0.0) & (raw < 1.0), raw, np.nan)
    if encoding == "reversed":
        return near / d if np.isinf(far) else near * far / (near + d * (far - near))
    if encoding == "Vulkan":
        return near / (1.0 - d) if np.isinf(far) else far * near / (far - d * (far - near))
    ndc = 2.0 * d - 1.0
    if np.isinf(far):
        return 2.0 * near / (1.0 - ndc)
    return 2.0 * far * near / ((far + near) - ndc * (far - near))


def unproject(view, uv, z):
    q, p, fov = view
    l, r, u, d = np.tan(fov)
    tx = l + uv[..., 0] * (r - l)
    ty = u - uv[..., 1] * (u - d)
    local = np.stack([tx * z, ty * z, -z], axis=-1)
    return np.array([quat_rotate(q, v) for v in local.reshape(-1, 3)]).reshape(local.shape) + p


def project(view, points):
    q, p, fov = view
    qi = np.array([-q[0], -q[1], -q[2], q[3]])
    local = np.array([quat_rotate(qi, v - p) for v in points.reshape(-1, 3)]).reshape(points.shape)
    l, r, u, d = np.tan(fov)
    z = -local[..., 2]
    tx, ty = local[..., 0] / z, local[..., 1] / z
    return np.stack([(tx - l) / (r - l), (u - ty) / (u - d)], axis=-1), z


def main():
    meta, views, raw = parse(sys.argv[1])
    w, h = meta["w"], meta["h"]
    print(f"grid {w}x{h}, near {meta['near']}, far {meta['far']}")
    xs, ys = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    for encoding in ENCODINGS:
        z = [linear(raw[l], meta["near"], meta["far"], encoding) for l in range(2)]
        uv = np.stack([xs, ys], axis=-1)
        points = unproject(views[0], uv, z[0])
        uv1, ahead = project(views[1], points)
        inside = np.all((uv1 >= 0) & (uv1 < 1), axis=-1) & np.isfinite(ahead)
        ix = np.clip((uv1[..., 0] * w).astype(int), 0, w - 1)
        iy = np.clip((uv1[..., 1] * h).astype(int), 0, h - 1)
        sensed = z[1][iy, ix]
        # Relative: a decoding that shrinks everything shrinks its errors.
        err = (np.abs(sensed - ahead) / ahead)[inside & np.isfinite(sensed)]
        median = np.nanmedian(z[0])
        print(
            f"{encoding:9} median {median:5.2f} m; eyes disagree by {np.mean(err) * 100:5.1f}% "
            f"on average, {np.percentile(err, 90) * 100:5.1f}% at the 90th percentile"
        )
    z = linear(raw[0], meta["near"], meta["far"], "OpenGL")
    print("left eye, metres (OpenGL depth), every 4th sample:")
    for row in z[::4, ::4]:
        print(" ".join(f"{v:5.2f}" for v in row))


if __name__ == "__main__":
    main()
