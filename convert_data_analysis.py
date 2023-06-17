import json
from collections import Counter

total = json.loads(open('/paddle/dataset/nfdw/total.json').read())
Counter([x['category_id'] for x in total['annotations']])
# Counter({1: 540, 3: 88, 2: 103, 4: 76})

Counter([x['image_id'] for x in total['annotations']])
