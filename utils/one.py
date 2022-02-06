import os, sys
from numpy import mat
from pymatgen.ext.matproj import MPRester
import json

API_TOKEN = os.environ.get("MPTOKEN")

mpr = MPRester(API_TOKEN)

crt = {
    'elements': {'$all': ['N']},
    'nelements': 3,
    'band_gap': {'$lt': 1},
    'anonymous_formula': {"A": 1, "B": 1, "C": 3}
}
prt = ['material_id', 'pretty_formula', 'band_gap', 'unit_cell_formula']

materials = mpr.query(criteria=crt, properties=prt)
# materials = mpr.query(criteria="**N4", properties=['material_id', 'pretty_formula', 'band_gap'])


# metal_materials = [material for material in materials if material['band_gap'] <= 0]

for i in materials: print(i)

print(f"Number of materials : {len(materials)}")

## Save the data to file
# with open('TMR_Data.json', 'w') as fo:
#     json.dump({'Total materials': len(materials), "Materials": materials}, fo, indent=2)


















