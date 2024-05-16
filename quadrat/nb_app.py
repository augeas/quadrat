
from . import quadtorch

def inflate_seq(name, size, n, pitch, roll, cols):
    bias, coeff = quadtorch.str2coeffs([name])

    trial = quadtorch.attractor_trial(bias, coeff)

    seq_bias, seq_coeff, perturb, _ = quadtorch.perturb_3d(bias, coeff, size, n, pitch, roll, cols)

    points = quadtorch.attractor_points(seq_bias, seq_coeff)

    return {
        'name': name,
        'bias': seq_bias, 'coeff': seq_coeff, 'points': points,
        'dim': trial[-2], 'lyap': trial[-1],
        'meta': {
            'size': size, 'n': n, 'pitch': pitch, 'roll': roll,
            'cols': cols
        }
    }





