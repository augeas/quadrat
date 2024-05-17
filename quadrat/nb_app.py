
import torch as tc

from . import quadtorch


def inflate_seq(name, size, n, pitch, roll, cols):
    bias, coeff = quadtorch.str2coeffs([name])

    trial = quadtorch.attractor_trial(bias, coeff)

    seq_bias, seq_coeff, perturb, _ = quadtorch.perturb_3d(bias, coeff, size, n, pitch, roll, cols)

    points = quadtorch.attractor_points(seq_bias, seq_coeff)

    return {
        'name': name,
        'bias': seq_bias, 'coeff': seq_coeff, 'points': points,
        'dim': float(trial[-2][0]), 'lyap': float(trial[-1][0]),
        'meta': {
            'size': size, 'n': n, 'pitch': pitch, 'roll': roll,
            'cols': cols
        }
    }

def inflate_img(name, size=500, points=200000):
    bias, coeff = quadtorch.str2coeffs([name])
    shape = (size, size)
    trial = quadtorch.attractor_trial(bias, coeff)
    points = quadtorch.attractor_points(
        bias, coeff, n_points=points, prev_points=trial[2]
    )
    img, minima, ranges = attractor_img(points, shape, common=False)
    seq = img_seq(points, shape, minima, ranges)

    return {
        'name': name, 'img': img, 'seq': seq,
        'dim': float(trial[-2][0]), 'lyap': float(trial[-1][0])
    }

def render_track(inflated_img, fft_size=8192, hop_size=2048, fft_points=256)
    window = tc.blackman_window(fft_size)

    segments = tc.zeros(fft_points, 1, fft_size, 2)

    segments[:, :, :, 0] = fft_segments(
        spectra_seq(
            inflated_img['img'],
            inflated_img['seq']
        ), window
    )
    segments[:, :, :, 1] = fft_segments(
        spectra_seq(
            inflated_img['img'].mT,
            inflated_img['seq']
        ), window
    )

    name = inflate_img['name']
    os.mkdir(name)

    wavfile.write(
        name.join(('./', '/','.wav')), 44100,
        stagger_segments(segments, hop_size).numpy()
    )


