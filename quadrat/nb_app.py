
import os
import subprocess

import ipywidgets as widgets
from IPython.display import display, Audio, FileLink
import numpy as np
from scipy.io import wavfile
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
    if 0 in trial[2].shape:
        prev = None
    else:
        prev = trial[2]
    points = quadtorch.attractor_points(
        bias, coeff, n_points=points, prev_points=prev
    )
    img, minima, ranges = quadtorch.attractor_img(points, shape, common=False)
    seq = quadtorch.img_seq(points, shape, minima, ranges)

    return {
        'name': name, 'img': img, 'seq': seq,
        'dim': float(trial[-2][0]), 'lyap': float(trial[-1][0])
    }

__ffmpeg_args__ = [
    'ffmpeg',
    '-y',
    '-i',
    'NAME/NAME.wav',
    '-i',
    'NAME/NAME.png',
    '-map',
    '0:0',
    '-map',
    '1:0',
    '-id3v2_version',
    '3',
    '-metadata:s:v',
    'title="Album cover"',
    '-metadata:s:v',
    'comment="Cover (front)"',
    'NAME/NAME.mp3'
]

def render_track(inflated_img, fft_size=8192, hop_size=2048, fft_points=512):
    window = tc.blackman_window(fft_size)

    segments = tc.zeros(fft_points, 1, fft_size, 2)

    segments[:, :, :, 0] = quadtorch.fft_segments(
        quadtorch.spectra_seq(
            inflated_img['img'],
            inflated_img['seq'][:fft_points]
        ), window
    )
    segments[:, :, :, 1] = quadtorch.fft_segments(
        quadtorch.spectra_seq(
            inflated_img['img'].mT,
            inflated_img['seq'][:fft_points]
        ), window
    )

    name = inflated_img['name']
    try:
        os.mkdir(name)
    except:
        pass

    audio = quadtorch.stagger_segments(segments, hop_size).numpy()

    fname = name.join(('./', '/','.wav'))

    wavfile.write(
        name.join(('./', '/','.wav')), 44100, audio
    )

    proc_args = __ffmpeg_args__.copy()
    for arg in (3, 5, -1):
        proc_args[arg] = proc_args[arg].replace('NAME', name)
    out = subprocess.run(
        proc_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )

    #return ((0.5 + audio.T / 2) * 65535).astype(np.uint16)
    return proc_args[-1]

class SingleImageApp(object):
    def __init__(self, name=None, size=600, fft_size=8192, hop_fraction=4,
        fft_points=1024, player=False):

        self.image_name = widgets.HTML()
        self.search_button = widgets.Button(description='new image')
        self.search_button.on_click(self.new_image)
        self.image_id_box = widgets.VBox([self.image_name, self.search_button])
        self.image_meta = widgets.HTML(layout = widgets.Layout(width='400px'))
        self.image_info = widgets.HBox([
            self.image_id_box, self.image_meta
        ])

        self.image_box = widgets.Output()
        self.fft_size_box = widgets.Dropdown(
            options=[1024, 2048, 4096, 8192, 16384],
            value=fft_size, description='fft size:',
            layout = widgets.Layout(width='150px')
        )
        self.hop_size_box = widgets.Dropdown(
            options=[('1/2', 2), ('1/4', 4), ('1/8', 8), ('1/16', 16)],
            value=hop_fraction, description='hop size:',
            layout = widgets.Layout(width='150px')
        )
        self.points_box = widgets.Dropdown(
            options = [256, 512, 1024, 2048, 4096],
            value = fft_points, description = 'points:',
            layout = widgets.Layout(width='150px')
        )
        self.audio_button = widgets.Button(description='build audio')
        self.audio_button.on_click(self.build_audio)
        self.audio_inputs_box = widgets.HBox([
            self.fft_size_box, self.hop_size_box, self.points_box, self.audio_button
        ])

        self.player = player
        self.audio_download_box = widgets.Output()
        self.audio_play_box = widgets.Output()
        self.audio_outputs_box = widgets.HBox([
            self.audio_play_box, self.audio_download_box
        ])

        self.box = widgets.VBox([
            self.image_info, self.audio_inputs_box, self.audio_outputs_box, self.image_box
        ])

        self.name = name
        self.image_size = size
        self.image_points = int(size * size * 0.625)
        self.image = None
        self.search = quadtorch.iter_search(100)
        self.audio_file = None

    def update_image(self):
        self.image_name.value = '<h2>{}</h2>'.format(self.name)
        if self.image is None:
            self.image = inflate_img(
                self.name, size=self.image_size, points=self.image_points
            )
        self.image_meta.value = '''
            <table>
                <tr>
                    <td>correlation dimension:</td><td>{:1.3f}</td>
                </tr>
                <tr>
                    <td>Lyapunov exponent</td><td>{:1.3f}</td>
                </tr>
            </table>
        '''.format(self.image['dim'], self.image['lyap'])

        self.image_box.clear_output()
        img = quadtorch.render_img(
            self.image['img'].reshape(
                self.image_size, self.image_size
            )
        )
        with self.image_box:
            display(img)
        try:
            os.mkdir(self.name)
        except:
            pass
        img.save('/'.join((self.name, '{}.png'.format(self.name))))

    def new_image(self, _):
        self.toggle_controls()
        self.image_name.value = '<h2>searching</h2>'
        search_result = self.search.__next__()
        self.name = search_result['name']
        points = quadtorch.attractor_points(
            search_result['bias'].reshape(1, 2, 1),
            search_result['coeff'].reshape(1, 2, 5),
            n_points = self.image_points,
            prev_points = search_result['points']
        )
        shape = (self.image_size, self.image_size)
        img, minima, ranges = quadtorch.attractor_img(
            points, shape, common=False
        )
        seq = quadtorch.img_seq(points, shape, minima, ranges)

        self.image = {
            'name': self.name, 'img': img, 'seq': seq,
            'dim': search_result['dim'].item(),
            'lyap': search_result['lyap'].item()
        }
        self.update_image()
        self.toggle_controls(False)

    def toggle_controls(self, toggle=True):
        for cntrl in (
            self.audio_button, self.fft_size_box, self.hop_size_box,
            self.points_box, self.search_button
        ):
            cntrl.disabled = toggle

    def build_audio(self, _):
        self.toggle_controls()
        self.audio_download_box.clear_output()
        if self.player:
            self.audio_play_box.clear_output()
        self.audio_file = render_track(self.image,
            fft_size = self.fft_size_box.value,
            hop_size = self.fft_size_box.value // self.hop_size_box.value,
            fft_points = self.points_box.value
        )
        with self.audio_download_box:
            display(FileLink(self.audio_file))
        if self.player:
            with self.audio_play_box:
                display(Audio(filename=self.audio_file, rate=441000))
        self.toggle_controls(False)

    def show(self, auto=True):
        display(self.box)
        if self.name:
            self.toggle_controls()
            self.update_image()
            if auto:
                self.build_audio(None)
            self.toggle_controls(False)
        else:
            self.toggle_controls()
            self.new_image(None)
            if auto:
                self.build_audio(None)
            self.toggle_controls(False)
