import os, json

with open("PubChemElements_all.json") as fo:
    data = json.load(fo)
    

all_elements = []
for j in data['Table']['Row']:
    
    element_obj = {}
    for element in list(zip(data['Table']['Columns']['Column'], j['Cell'])):
        
        property = element[0]
        value = element[1]
        
        element_obj[property] = value
        
    all_elements.append(element_obj)

with open("my_elements_data.json", 'w') as fo:
    json.dump(all_elements, fo, indent=2)

    