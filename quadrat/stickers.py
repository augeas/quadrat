
import io
from itertools import chain
import os

from PIL import Image, ImageDraw, ImageFont
import segno

from .nb_app import inflate_img, inflate_search
from .quadtorch import render_img

__font_path__ = '/'.join(__file__.split('/')[:-1]) + '/fonts'

__fonts__ = {
    fnt.rstrip('.ttf'): '/'.join((__font_path__, fnt))
    for fnt in os.listdir(__font_path__)
}

__url_templates__ = {
    'binder_image': (
        'https://mybinder.org/v2/gh/augeas/quadrat/main?'
         + 'urlpath=notebooks%2Fimage.ipynb%3Fname%3D%22{}%22%26autorun%3Dtrue'
    )
}

def_prefix = [
    'AGGRORHYTHMIC',
    'COMPOSITION',
    'WHAT SHOULD A'
]

def_suffix = [
    'SOUND LIKE?'
]

def expand_text(text, font_path, width, size=5, incr=5, margin=50):
    font_size = size
    box = (0, 0, 0, 0)
    new_size = font_size + incr
    font = ImageFont.truetype(font_path, size=new_size)
    new_box = font.getbbox(text)
    while new_box[2] + margin < width:
        font_size = new_size
        box = new_box
        new_size = font_size + incr
        font = ImageFont.truetype(font_path, size=new_size)
        new_box = font.getbbox(text)
    return (font_size,) + box

def centre(box, width):
    return (width // 2) - box[3] // 2

def qr_sticker(name, prefix, suffix, font_path, width=600, margin=40, dest='binder_image'):
    inflated = inflate_img(name, size=width, points=int(0.5 * width * width))
    if inflated is None:
        return None
    attractor_img = render_img(inflated['img'].reshape(width, width), colour='b&w')
    url = __url_templates__[dest].format(name)
    code = segno.make(url)
    scale = width // (code._matrix_size[0] + 2 * code.default_border_size)
    out = io.BytesIO()
    code.save(out, scale=scale, kind='png')
    out.seek(0)
    code_img = Image.open(out)
    code_height = code_img.size[1]
    prefix_dims = [expand_text(txt, font_path, width) for txt in prefix]
    suffix_dims = [expand_text(txt, font_path, width) for txt in suffix]
    name_dims = expand_text(name, font_path, width)
    prefix_height = sum([dim[4] for dim in prefix_dims])
    suffix_height = sum([dim[4] for dim in suffix_dims])
    text_height = prefix_height + name_dims[4] + suffix_height
    total_height = text_height + code_height + width
    image = Image.new("RGB", (width + margin, total_height), "white")
    draw = ImageDraw.Draw(image)
    y_off = 0
    for txt, dim in zip(prefix, prefix_dims):
        font = ImageFont.truetype(font_path, size=dim[0])
        x_off = centre(dim, width)
        draw.text((margin + x_off, y_off), txt, fill='black', font=font)
        y_off += dim[4]
    x_off = centre(name_dims, width)
    image.paste(attractor_img, (margin, y_off))
    y_off += width
    draw.text((margin + x_off, y_off), name, fill='black', font=font)
    y_off += name_dims[4]
    for txt, dim in zip(suffix, suffix_dims):
        font = ImageFont.truetype(font_path, size=dim[0])
        x_off = centre(dim, width)
        draw.text((margin + x_off, y_off), txt, fill='black', font=font)
        y_off += dim[4]
    image.paste(code_img, (margin, y_off))
    return image

def empty_corners(image, w, h):
    img = image.T
    corners = {
        'bottom_left': img[0:w, -h:],
        'bottom_right': img[-w:, -h:],
        'top_left': img[0:w, 0:h],
        'top_right': img[-w:, 0:h]
    }
    return [pos for pos, box in corners.items() if box.sum() == 0]

def sticker(res, size=282, border=26, colour='green', font='Swansea-q3pd', font_size=12):
    im_size = size - 2 * border
    image = inflate_search(res, im_size, int(0.5 * im_size * im_size), sequence=False)
    im_font = ImageFont.truetype(__fonts__[font], size=font_size)
    w, h = im_font.getbbox(image['name'])[-2:]
    pad_w = w + 2
    pad_h = h + 2
    pos = empty_corners(image['img'], pad_w, pad_h)
    if not pos:
        return None
    if pos[0].startswith('bottom'):
        y_off = im_size - h - 1
    else:
        y_off = 1
    if pos[0].endswith('left'):
        x_off = 1
    else:
        x_off = im_size - w - 1
    rendered = render_img(image['img'], colour=colour)
    if colour in ('red', 'green', 'blue', 'cyan'):
        fill = colour
    draw = ImageDraw.Draw(rendered)
    draw.text((x_off, y_off), image['name'], fill=fill, font=im_font)
    final = Image.new("RGB", (size, size), 'black')
    final.paste(rendered, (border, border))
    return final


