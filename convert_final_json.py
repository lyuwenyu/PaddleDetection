import json

data = json.loads(open('bbox.json', 'r').read())

_data = []
for blob in data:
    if blob['score'] > 0.5:
        _data.append(blob)

with open('bbox_final.json', 'w') as f:
    output_json = json.dumps(_data)
    f.write(output_json)

_data = []
for blob in data:
    if blob['score'] > 0.6:
        _data.append(blob)
