import colorsys, sys
from pymol import cmd
from pymol.cgo import *

def draw_links(selection1="(pk1)", selection2="(pk2)", color=None, color2=None, radius=None, object_name=None):
  """
  AUTHOR
    Robert L. Campbell

    Draw links between C-alpha carbons in two selections.

  USAGE
    draw_links(selection2='sel', selection2='sel', color='red', color2='None', radius=None, object_name=None)

    - color defaults to red and color2 defaults to be the same as color
    - color and color2 can be specified by name, or by color tuple with values
      between 0 and 1, e.g. red is (1,0,0) and magenta is (1,0,1)
    - radius defaults to 0.2A
    - object_name defaults to 'link'.
  """
  # Set defaults
  if not color:
    tup_color = [1., 0., 0.]

  if isinstance(color, str):
    try:
      tup_color = list(map(float, color.replace('(', '').replace(')', '').split(',')))
    except ValueError:
      tup_color = list(cmd.get_color_tuple(color))
  elif isinstance(color, (list, tuple)):
    tup_color = list(color)

  if not color2:
    tup_color2 = tup_color
  if isinstance(color2, str):
    try:
      tup_color2 = list(map(float, color2.replace('(', '').replace(')', '').split(',')))
    except ValueError:
      tup_color2 = list(cmd.get_color_tuple(color2))
  elif isinstance(color2, (list, tuple)):
    tup_color2 = list(color2)

  if not radius:
    radius = 0.2
  else:
    radius = float(radius)

  # Check if pk1 and pk2 are requested and defined
  if ((selection1 == '(pk1)' and 'pk1' not in cmd.get_names("selections")) or
      (selection2 == '(pk2)' and 'pk2' not in cmd.get_names("selections"))):
    print("You must pick the atoms with 'pk1' and 'pk2' or specify a selection")
    print(cmd.get_names("selections"))
    sys.exit(2)
  if not selection1 or not selection2:
    print("You must enter two residue selections")
    sys.exit(1)

  m1 = cmd.get_model(selection1)
  m2 = cmd.get_model(selection2)
  coord1 = []
  coord2 = []

  if len(m1.atom) == 0:
    print("Sorry, no atoms selected in selection 1:", selection1)
    sys.exit(1)
  elif len(m2.atom) == 0:
    print("Sorry, no atoms selected in selection 2:", selection2)
    sys.exit(1)
  else:
    if len(m1.atom) > 1 or len(m2.atom) > 1:
      for a in m1.atom:
        if a.name == 'CA':
          coord1.append(a.coord)

      for b in m2.atom:
        if b.name == 'CA':
          coord2.append(b.coord)
    else:
      coord1 = (m1.atom[0].coord,)
      coord2 = (m2.atom[0].coord,)

  cyl_obj = []
  for x, y in zip(coord1, coord2):
    cyl_obj.extend([CYLINDER] + x + y + [radius] + tup_color + tup_color2)

  if not object_name:
    if "link" not in cmd.get_names("objects"):
      object_name = "link"
    else:
      count = 0
      for n in cmd.get_names("objects"):
        if n.startswith("link"):
          count += 1

      object_name = "link" + str(count)
  else:
    object_name = object_name

  cmd.load_cgo(cyl_obj, object_name)

cmd.extend("draw_links", draw_links)

