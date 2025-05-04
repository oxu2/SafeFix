import json
with open('exampleData/celebA/attribute.json') as f:
    attrs = json.load(f)

if 'color' in attrs:
    print('Color uncertain count:', len(attrs['color']['uncertain_idx']))
if 'texture' in attrs:
    print('Texture uncertain count:', len(attrs['texture']['uncertain_idx']))
if 'skin' in attrs:
    print('Skin uncertain count:', len(attrs['skin']['uncertain_idx']))