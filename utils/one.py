import os, sys
from pymatgen.ext.matproj import MPRester

API_TOKEN = os.environ.get("MPTOKEN")

mpr = MPRester(API_TOKEN)

crt = {
    'elements': {'$all': ['N']},
    'nelements': 3,
    'band_gap': {'$lte': 0}
}
prt = ['material_id', 'pretty_formula', 'band_gap', 'unit_cell_formula']

materials = mpr.query(criteria=crt, properties=prt)

# metal_materials = [material for material in materials if material['band_gap'] <= 0]

for i in materials: print(i)

print(f"Number of materials : {len(materials)}")
