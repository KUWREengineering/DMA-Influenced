import os
import csv
from bs4 import BeautifulSoup

DATA_DIR = 'data/valve-html'
VTYPE = ['?', 'BV', 'GV', 'RCV']
VSTATUS = ['ปกติ', 'รั่ว', 'จม', 'ชำรุด', 'ซ่อม']

with open('data/valves.csv', 'w') as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(['vid', 'percent', 'rounds', 'nickname', 'type', 'status'])
    for fname in os.listdir(DATA_DIR):
        with open(os.path.join(DATA_DIR, fname)) as f:
            soup = BeautifulSoup(f, 'html.parser')
        vid = soup.find(id='num1').attrs.get('value', '')
        percent = soup.find(id='default_df').attrs.get('value', '')
        rnd = soup.find(id='default_rnd').attrs.get('value', '')
        name = soup.find(id='nickname').attrs.get('value', '')
        brand = soup.find(id='default_brd').attrs.get('value', '')
        size = soup.find(id='default_size').attrs.get('value', '')
        depth = soup.find(id='deep').attrs.get('value', '')

        vtype = []
        for i in [0,1,2,3]:
            vtype.append('checked' in soup.find(id=f'vtype{i}').attrs)

        vstatus = []
        for i in [1,2,3,4,5]:
            vstatus.append('checked' in soup.find(id=f'rad{i}').attrs)

        types = ','.join([type for type,checked in zip(VTYPE, vtype) if checked])
        statuses = ','.join([status for status,checked in zip(VSTATUS, vstatus) if checked])

        writer.writerow([vid, percent, rnd, name, types, statuses])
