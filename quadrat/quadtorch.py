
import json
import logging
import os
import random

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import wavfile
import torch as tc


__VALID_CHARS__ = list(map(chr, range(65, 65+25)))

def init_var(size, x=0.05, y=0.05):
    var = tc.zeros(size, 2, 1, dtype=tc.float64)
    var[:, 0] = x
    var[:, 1] = y
    return var

def var_vec(var):
    size = var.shape[0]
    vec = tc.zeros(size, 5, 1)
    vec[:, 1:, :] = (var * var.reshape(size, 1, 2)).reshape(size, 4, 1)
    vec[:, 0, :] = var[:, 0]
    vec[:, 3, :] = var[:, 1]
    return vec

def quad_iterate(var, bias, coeff):
    size = var.shape[0]
    vec = tc.zeros(size, 5, 1, dtype=tc.float64)
    vec[:, 1:, :] = (var * var.reshape(size, 1, 2)).reshape(size, 4, 1)
    vec[:, 0, :] = var[:, 0]
    vec[:, 3, :] = var[:, 1]

    return bias + coeff @ vec

def str2coeffs(names):
    dim = len(names)
    coeff_range = tc.arange(-1.2, 1.3, 0.1, dtype=tc.float64)
    bias_coeff = tc.zeros(dim, 12, dtype=tc.float64)
    for i, name in enumerate(names):
        bias_coeff[i, :] = coeff_range[[ord(c) - 65 for c in name.upper()]]
    bias_coeff = bias_coeff.reshape(dim, 2, 6)

    bias = bias_coeff[:, :, 0].reshape(dim, 2, 1)
    coeffs = bias_coeff[:, :, 1:]

    return bias, coeffs

N2 = 0.5 * 0.5
N1 = 0.05 * 0.05

def attractor_trial(bias, coeffs, n_points=5000, skip=500, dist=0.0001):
    total_points = skip + n_points
    n_attractors = coeffs.shape[0]
    delta = np.sqrt(dist / 2.0)
    var = init_var(n_attractors)
    displace = init_var(n_attractors, x=delta, y=delta)

    points = tc.zeros(total_points, n_attractors, 2, 1, dtype=tc.float64)
    moved = tc.zeros(total_points, n_attractors, dtype=tc.float64)
    bounded = tc.ones(n_attractors, dtype=tc.bool)

    for i in range(total_points):
        idx = tc.where(bounded)[0]
        n_bound = bounded.sum()

        if n_bound == 0:
            break

        displaced_var = quad_iterate(var[idx] + displace[idx], bias[idx], coeffs[idx])
        var[idx] = quad_iterate(var[idx], bias[idx], coeffs[idx])

        displace[idx] = var[idx] - displaced_var
        moved[i, idx] = displace[idx].square().sum(axis=1).sqrt().reshape(-1)
        displace[idx] *= (dist / moved[i, idx]).reshape(n_bound, 1, 1)

        points[i, idx, :, :] = var[idx]
        bounded[idx] = (var[idx].abs().amax(dim=1) < 1e6).reshape(-1)

    idx = tc.where(bounded)[0]
    att_points = points[skip:, idx]

    disp_sq = (att_points - att_points[tc.randperm(n_points)]).square().sum(axis=2)
    dim = tc.log10((disp_sq < N2).sum(dim=0) / (disp_sq < N1).sum(dim=0)).flatten()

    lyap = (tc.log2(moved[skip:, idx] / dist).sum(axis=0) / n_points)

    return bias[idx], coeffs[idx], att_points, dim, lyap

def attractor_points(bias, coeffs, n_points=5000, prev_points=None):
    n_attractors = coeffs.shape[0]
    points = tc.zeros(n_points, n_attractors, 2, 1, dtype=tc.float64)
    if prev_points is None:
        var = init_var(n_attractors)
        offset = 0
    else:
        var = prev_points[-1]
        offset = prev_points.shape[0]
        points[0:offset] = prev_points
    points[offset] = quad_iterate(var, bias, coeffs)
    for i in range(offset + 1, n_points):
        points[i] = quad_iterate(points[i-1], bias, coeffs)
    return points

def img_hist(points, shape, ranges, img):
    img[:, :], _ = tc.histogramdd(points, bins=shape, range=ranges, density=False)

def attractor_img(points, shape, common=True):
    n_points, n_attract = points.shape[0:2]
    images = tc.zeros(n_attract, *shape)
    minima = points.amin(dim=0).reshape(n_attract, 2)
    maxima = points.amax(dim=0).reshape(n_attract, 2)
    if common:
        minima = minima.amin(dim=0).repeat(n_attract, 1).reshape(n_attract, 2)
        maxima = maxima.amax(dim=0).repeat(n_attract, 1).reshape(n_attract, 2)
    flat_ranges = tc.zeros(n_attract, 4)
    flat_ranges[:, 0::2] = minima
    flat_ranges[:, 1::2] = maxima

    point_ranges = maxima - minima

    for i in range(n_attract):
        img_hist(points[:, i].reshape(n_points, 2), shape, flat_ranges[i].tolist(), images[i])

    images = images / images.amax(dim=(1, 2)).reshape(n_attract, 1, 1)

    return images, minima, point_ranges

def new_name():
    return ''.join(random.choice(__VALID_CHARS__) for _ in range(12))

def is_aesthetic(dim, lyap):
    return tc.logical_and(
        tc.logical_and(dim > 1.1, dim < 1.5),
        tc.logical_and(lyap > 0.01, lyap < 0.3)
    )

def search_batch(size):
    names = [new_name() for _ in range(size)]
    bias, coeff = str2coeffs(names)
    bias, coeff, points, dim, lyap = attractor_trial(bias, coeff)
    pretty = tc.where(is_aesthetic(dim, lyap))[0]
    found = [names[i] for i in pretty]
    return (found, bias[pretty], coeff[pretty], points[:, pretty],
        dim[pretty], lyap[pretty])

def shell_grid(size, n, dim=2, inner=0.75):
    ls = tc.linspace(-size, size, n)
    mg = tc.meshgrid(*[ls for i in range(dim)], indexing='ij')
    r = tc.add(*map(tc.square, mg))
    size_sq = size * size
    include =  tc.logical_and(r <= size_sq, r >= inner * inner * size_sq)
    return tc.stack([grid[include] for grid in mg]).T

def perturb(bias, coeffs, size, n):
    sg = shell_grid(size, n, 2)
    npoints, _ = sg.shape

    candidates = tc.zeros(npoints, 12, dtype=tc.float64)
    candidates[:, 0:2] = bias.reshape(2)
    candidates[:, 2:] = coeffs.reshape(10)

    cols = np.random.choice(np.arange(12), 2, replace=False)
    candidates[:, cols] += sg

    bias_candidates = candidates[:, 0:2].reshape(npoints, 2, 1)
    coeff_candidates = candidates[:, 2:].reshape(npoints, 2, 5)

    return bias_candidates, coeff_candidates, sg

def perturb_3d(bias, coeff, size, n, pitch=None, roll=None, cols=None):

    candidates = tc.zeros(n, 12, dtype=tc.float64)
    candidates[:, 0:2] = bias.reshape(2)
    candidates[:, 2:] = coeff.reshape(10)

    if pitch is None:
        pitch = tc.pi * (1 - 2 * tc.rand(1))
    else:
        pitch = tc.tensor(pitch)
    if roll is None:
        roll = tc.pi * (1 - 2 * tc.rand(1))
    else:
        roll = tc.tensor(roll)

    sin_pitch = tc.sin(pitch)
    cos_pitch = tc.cos(pitch)

    pitch_mat = tc.zeros((3, 3))
    pitch_mat[0, 0] = cos_pitch
    pitch_mat[0, 2] = sin_pitch
    pitch_mat[1, 1] = 1.0
    pitch_mat[2, 0] = -sin_pitch
    pitch_mat[2, 2] = cos_pitch

    cos_roll = tc.cos(roll)
    sin_roll = tc.sin(roll)

    roll_mat = tc.zeros((3, 3))
    roll_mat[0, 0] = 1.0
    roll_mat[1, 1] = cos_roll
    roll_mat[1, 2] = -sin_roll
    roll_mat[2, 1] = sin_roll
    roll_mat[2, 2] = cos_roll

    rot_mat = pitch_mat @ roll_mat

    theta = tc.linspace(-tc.pi, tc.pi, n+1)[:-1]

    grid = tc.zeros((3 , n))
    grid[0, :] = size * tc.sin(theta)
    grid[1, :] = size * tc.cos(theta)

    perturb = (rot_mat @ grid).T
    if cols is None:
        cols = tc.from_numpy(
            np.random.choice(np.arange(12), 3, replace=False)
        )
    else:
        cols = tc.tensor(cols)
    candidates[:, cols] += perturb

    bias_candidates = candidates[:, 0:2].reshape(n, 2, 1)
    coeff_candidates = candidates[:, 2:].reshape(n, 2, 5)

    meta = {
        'size': size, 'n': n,
        'pitch': float(pitch), 'roll': float(roll),
        'cols': cols.tolist()
    }

    return bias_candidates, coeff_candidates, perturb, meta

def dist_tab(points):
    npoints, _ = points.shape
    tab = tc.zeros(npoints, npoints)
    for i in range(npoints):
        for j in range(i):
            tab[i, j] = np.sqrt(tc.square(points[i] - points[j]).sum())
            tab[j, i] = tab[i,j]
    scale = 100 / tab.max()

    return tab * scale

def path_dist(perm, dist):
    return dist[perm, tc.roll(perm, 1)]

def eval_seq(bias, coeffs, grid, factor=3.0, three_d=True):
    bias, coeffs, points, dim, lyap = attractor_trial(bias, coeffs)
    pretty = tc.where(is_aesthetic(dim, lyap))[0]
    grid_points = grid[pretty]
    if three_d:
        path = tc.argsort(tc.atan2(grid_points[:, 1], grid_points[:, 0]))
    else:
        path = tc.arange(grid_points.shape[0])
    distance_table = dist_tab(grid_points)
    path_distances = path_dist(path, distance_table)
    accepted = (path_distances.max() / path_distances.min()) < factor

    if accepted:
        return (
            bias[pretty][[path]],
            coeffs[pretty][[path]],
            points[:, pretty][:, path]
        )
    else:
        return None

def img_seq(points, shape, minima, ranges):
    n_attract = points.shape[1]
    shape_vec = tc.tensor(shape).reshape(2, 1) - 1
    return (
        (points - minima.reshape(n_attract, 2, 1))
        / ranges.reshape(n_attract, 2, 1) * shape_vec
    ).int()

def spectra_seq(img, seq):
    n_img = img.shape[0]
    img_size = img.shape[-1]
    n_points = seq.shape[0]
    spectra = tc.zeros(n_points, n_img, 2, img_size)
    for i in range(n_img):
        x = seq[:,i,0,0]
        y = seq[:,i,1,0]
        spectra[:,i,0,:] = img[i,x,:]
        spectra[:,i,1,:] = img[i,:,y].T
    return spectra

def fft_segments(spectra, window):
    n_points, n_images = spectra.shape[0:2]
    spec_size = spectra.shape[-1]
    spec_plus = spec_size + 1
    fft_size, = window.shape

    cspec = tc.zeros(n_points, n_images, 2*spec_size+1, dtype=tc.complex64)
    cspec[:, :, 1:spec_plus].real = spectra[:, :, 0, :]
    cspec[:, :, spec_plus:].real = spectra[:, :, 0, :].flip(dims=(2,))
    cspec[:, :, 1:spec_plus].imag = spectra[:, :, 1, :]
    cspec[:, :, spec_plus:].imag = -spectra[:, :, 1, :].flip(dims=(2,))

    return tc.fft.ifft(cspec, n=fft_size).real * window

def stagger_segments(segs, hop, wrap=True):
    n_points, attractors, samples, channels = segs.shape
    n_segs = n_points * attractors
    overlap = samples - hop
    all_segs = segs.reshape(n_segs, samples, channels)
    total_samples = n_segs * hop + overlap

    if wrap:
        total_samples -= hop

    all_samples = tc.zeros(total_samples, channels)

    if wrap:
        seg_seq = all_segs[:-1]
        all_samples[0:hop] = all_segs[-1, -hop:, :]
        all_samples[-overlap:] = all_segs[-1, 0:overlap, :]
    else:
        seg_seq = all_segs

    for i, seg in enumerate(seg_seq):
        start = i * hop
        all_samples[start:start+samples] += seg

    s_min = all_samples.min()
    s_range = all_samples.max() - s_min

    return 2 * (all_samples - s_min) / s_range - 1

__primary__ = {
    'red': [0],
    'green': [1],
    'blue': [2],
    'cyan': [1, 2]
}
__cmaps__ = ('viridis', 'plasma', 'inferno', 'magma', 'twilight')

__segmented__ = ('hsv', 'coolwarm', 'cubehelix', 'brg')

def render_img(img, colour='green'):
    root_img = np.cbrt(img.numpy())
    if colour in __primary__.keys():
        rendered = np.zeros(img.shape + (3,), dtype=np.uint8)
        mono = (255 * root_img).astype(np.uint8)
        for rgb in __primary__[colour]:
            rendered[:, :, rgb] = mono
    elif colour == 'b&w':
        rendered = (255 - 255 * (root_img > 0)).astype(np.uint8)
    elif colour in __cmaps__:
        rgb = (255 * np.array(plt.get_cmap(colour).colors))
        rendered = rgb[(255 * root_img).astype(np.uint8)].astype(np.uint8)
    elif colour in __segmented__:
        col_map = plt.get_cmap(colour)
        rgb = (
            np.array(list(map(col_map, np.arange(col_map.N))))[:, :-1] * col_map.N
        ).astype(np.uint8)
        rendered = rgb[(col_map.N * root_img).astype(np.uint8)]
    else:
        rendered = (255 - 255 * root_img).astype(np.uint8)

    return Image.fromarray(rendered)

def iter_search(size=2000):
    while True:
        batch = search_batch(size)
        yield from (dict(
            zip(('name', 'bias', 'coeff', 'points', 'dim', 'lyap'), att)
        ) for att in zip(*batch))

def attractor_seq(n, grid_size=0.0075, grid_points=80, attempts=50, batch=2000):
    found = 0
    search = iter_search(batch)
    while found < n:
        candidate = search.__next__()
        name = candidate['name']
        print('TRYING: {}'.format(name))
        for i in range(attempts):
            bias, coeff, grid, meta = perturb_3d(
                candidate['bias'], candidate['coeff'],
                grid_size, grid_points
            )
            seq = eval_seq(bias, coeff, grid)
            if not seq is None:
                break
        if seq is None:
            print(
                'COULD NOT PERTURBATE {}'.format(name)
            )
        else:
            found += 1
            yield {
                'name': name,
                'bias': seq[0], 'coeff': seq[1], 'points': seq[2],
                'dim': candidate['dim'], 'lyap': candidate['lyap'],
                'meta': meta
            }



def render_video(
    seq, img_size=800, img_points=400000, fft_size=8192, hop_size=2048, fft_points=64, max_dur=None
):
    all_points = attractor_points(
        seq['bias'], seq['coeff'], n_points=img_points, prev_points=seq['points']
    )

    img_dim = (img_size, img_size)

    images, minima, ranges = attractor_img(all_points, img_dim)

    point_seqs = img_seq(all_points[0:fft_points], img_dim, minima, ranges)
    n_attract = point_seqs.shape[1]

    window = tc.blackman_window(fft_size)

    all_segs = tc.zeros(fft_points, n_attract, fft_size, 2)

    all_segs[:, :, :, 0] = fft_segments(spectra_seq(images, point_seqs), window)
    all_segs[:, :, :, 1] = fft_segments(spectra_seq(images.mT, point_seqs), window)

    name = seq['name']
    os.mkdir(name)

    wavfile.write(
        name.join(('./', '/','.wav')), 44100,
        stagger_segments(all_segs, hop_size).numpy()
    )

    for i, im in enumerate(images):
        img_fname = ('0000' + str(i))[-4:] + '.png'
        render_img(im).save('/'.join((name, img_fname)))

    with open('{}/{}.json'.format(name, name), 'w') as jfile:
        jfile.write(json.dumps({
            **{'name': name},
            **{k: seq[k].tolist() for k in ('bias', 'coeff')},
            **{k: float(seq.get(k)) for k in ('dim', 'lyap')},
            **seq['meta']
        }))

