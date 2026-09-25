"""Editable shared-prediction concept, drawn at its intended paper size."""
from pathlib import Path
from html import escape
import json
import math
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

OUT = Path(__file__).resolve().parent
W, H = 190, 150
INK, GRAY, TEAL, PALE = '#26343C', '#8C8C8C', '#1B8A8A', '#C5CCCF'
for role, file in [('regular', 'times.ttf'), ('italic', 'timesi.ttf'), ('bold', 'timesbd.ttf')]:
    pdfmetrics.registerFont(TTFont(role, str(Path('C:/Windows/Fonts') / file)))
pdf = canvas.Canvas(str(OUT / 'concept.pdf'), pagesize=(W, H))
pdf.setTitle('Same predictions, different targets')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}">',
       '<title>Same predictions, different targets</title>',
       '<desc>Schematic shared live state and candidate predictions. Gray final-goal scoring selects u1; teal observed-target scoring selects u2. The observed target is a recorded successor and is distinct from every predicted endpoint.</desc>']
objects = []


def text(name, x, y, value, size=8, color=INK, role='regular'):
    pdf.setFont(role, size)
    pdf.setFillColor(HexColor(color))
    pdf.drawString(x, H-y, value)
    style = (' font-style="italic"' if role == 'italic' else '') + (' font-weight="bold"' if role == 'bold' else '')
    svg.append(f'<text id="{name}" x="{x}" y="{y}" font-family="Times New Roman" font-size="{size}" fill="{color}"{style}>{escape(value)}</text>')
    objects.append(dict(id=name, kind='text', x=x, y=y, value=value, size=size, role=role))


def math_label(name, x, y, letter, subscript, color=INK, size=9):
    text(name, x, y, letter, size, color, 'italic')
    text(name+'-subscript', x+pdfmetrics.stringWidth(letter, 'italic', size), y+2,
         subscript, size*.73, color)


def line(name, points, color=INK, width=.7, dash=None):
    pdf.setStrokeColor(HexColor(color)); pdf.setLineWidth(width); pdf.setDash(dash or [])
    path = pdf.beginPath(); path.moveTo(points[0][0], H-points[0][1])
    for x, y in points[1:]: path.lineTo(x, H-y)
    pdf.drawPath(path, stroke=1, fill=0); pdf.setDash([])
    style = f' stroke-dasharray="{",".join(str(v) for v in dash)}"' if dash else ''
    coords = ' '.join(f'{x},{y}' for x, y in points)
    svg.append(f'<polyline id="{name}" points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"{style}/>')


def curve(name, points, color=INK, width=.9):
    a, b, c, d = points
    path = pdf.beginPath(); path.moveTo(a[0], H-a[1])
    path.curveTo(b[0], H-b[1], c[0], H-c[1], d[0], H-d[1])
    pdf.setStrokeColor(HexColor(color)); pdf.setLineWidth(width); pdf.setDash([])
    pdf.drawPath(path, stroke=1, fill=0)
    svg.append(f'<path id="{name}" d="M {a[0]} {a[1]} C {b[0]} {b[1]} {c[0]} {c[1]} {d[0]} {d[1]}" fill="none" stroke="{color}" stroke-width="{width}"/>')


def circle(name, x, y, r=1.9, fill=INK, stroke=None, width=.65):
    stroke = stroke or fill
    pdf.setFillColor(HexColor(fill)); pdf.setStrokeColor(HexColor(stroke)); pdf.setLineWidth(width)
    pdf.circle(x, H-y, r, fill=1, stroke=1)
    svg.append(f'<circle id="{name}" cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def rect(name, x, y, w, h, fill='#FFFFFF', stroke=PALE, width=.5):
    pdf.setFillColor(HexColor(fill)); pdf.setStrokeColor(HexColor(stroke)); pdf.setLineWidth(width)
    pdf.rect(x, H-y-h, w, h, fill=1, stroke=int(width>0))
    svg.append(f'<rect id="{name}" x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def arrow(name, points, color=INK, width=.6):
    line(name, points, color, width)
    (x0, y0), (x, y) = points[-2:]
    angle = math.atan2(y-y0, x-x0)
    wings = [(x-3*math.cos(angle-a), y-3*math.sin(angle-a)) for a in (-.48, .48)]
    line(name+'-head', [wings[0], (x, y), wings[1]], color, width)


def observation(name, x, y, dx, dy):
    rect(name+'-frame', x, y, 19, 19)
    rect(name+'-bar', x+4+dx, y+5+dy, 10, 3, '#789CAC', width=0)
    rect(name+'-stem', x+8+dx, y+8+dy, 3, 7, '#789CAC', width=0)
    circle(name+'-agent', x+4+dx, y+14+dy, 1.35, '#B99872')


# Both scoring rules use this one immutable set of candidate endpoints.
live = (14, 123)
ends = {'1': (127, 98), '2': (109, 72), '3': (124, 131)}
targets = {'final': (178, 98), 'observed': (123, 43)}
for name, desired in [('final', '1'), ('observed', '2')]:
    nearest = min(ends, key=lambda k: math.dist(ends[k], targets[name]))
    assert nearest == desired
    assert all(math.dist(end, targets[name]) > 1 for end in ends.values())

text('panel-title', 0, 9, '(a) Same predictions, different targets', role='bold')
text('recorded-label', 4, 23, 'Recorded')
observation('past', 4, 28, 0, 1)
observation('successor', 62, 28, 1, -1)
arrow('recorded-transition', [(26, 37.5), (58, 37.5)])
text('recorded-span', 29, 30, '5 steps')
arrow('successor-target', [(84, 37.5), (100, 37.5), (100, 43), (120, 43)], TEAL)

curve('candidate-u3', [live, (42, 135), (85, 127), ends['3']], PALE, .75)
curve('candidate-u1', [live, (47, 111), (77, 91), ends['1']], GRAY, 1.25)
curve('candidate-u2', [live, (44, 85), (73, 73), ends['2']], TEAL, 1.25)
line('final-scoring-distance', [ends['1'], targets['final']], GRAY, .85, [2, 2])
line('observed-scoring-distance', [ends['2'], targets['observed']], TEAL, .85, [2, 2])

circle('live', *live, 2.1)
circle('p1', *ends['1'], 1.9, GRAY)
circle('p2', *ends['2'], 1.9, TEAL)
circle('p3', *ends['3'], 1.9, '#FFFFFF', PALE)
circle('observed-target', *targets['observed'], 2.1, TEAL)
gx, gy = targets['final']
line('final-target', [(gx, gy-3), (gx+3, gy), (gx, gy+3), (gx-3, gy), (gx, gy-3)], GRAY, .8)

text('live-label', 1, 142, 'Live')
math_label('zt', 19, 142, 'z', 't')
math_label('p1-label', 130, 111, 'p', '1')
math_label('p2-label', 109, 85, 'p', '2')
math_label('p3-label', 132, 133, 'p', '3')
math_label('qobs', 128, 46, 'q', 'obs', TEAL)
math_label('zg', 174, 87, 'z', 'g', GRAY)
text('selected-u1-label', 59, 116, 'Select', color=GRAY)
math_label('selected-u1', 80, 116, 'u', '1', GRAY, 8)
text('selected-u2-label', 34, 75, 'Select', color=TEAL)
math_label('selected-u2', 55, 75, 'u', '2', TEAL, 8)

pdf.showPage(); pdf.save(); svg.append('</svg>')
(OUT/'concept.svg').write_text('\n'.join(svg), encoding='utf-8')
(OUT/'concept-objects.json').write_text(json.dumps(objects, indent=2))
(OUT/'concept-geometry.json').write_text(json.dumps(dict(live=live, endpoints=ends, targets=targets,
    nearest_endpoint={'final': '1', 'observed': '2'}, positions='schematic',
    width_points=W, height_points=H, base_font_points=8), indent=2))
print(json.dumps({'svg':str(OUT/'concept.svg'), 'pdf':str(OUT/'concept.pdf'), 'text_objects':len(objects)}))
