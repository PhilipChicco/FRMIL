import os 
from utils import get_files

TXT_ROOT = './datasets/wsi/cm16/splits/C16_TEST.txt'
assert os.path.isfile(TXT_ROOT) == True 

xml_files = get_files(TXT_ROOT, '.tif')

print(f'WSI FILES {len(xml_files)}')
SV_ROOT = './datasets/wsi/cm16/splits/'

with open(os.path.join(SV_ROOT,'C16_TEST_TUMOR.txt'), 'w') as f:
    count = 0
    for xml in xml_files:
        xml_name = os.path.split(xml)[-1].split('.')[0]
        if os.path.isfile(xml.replace('.tif', '.xml')):
            annot = xml.replace('.tif', '.xml')
            f.write(f'{xml}\n')
            f.write(f'{annot}\n')
            count += 1 
    print(f'TUMOR FILES : {count}')

with open(os.path.join(SV_ROOT,'C16_TEST_NORMAL.txt'), 'w') as f:
    count = 0
    for xml in xml_files:
        xml_name = os.path.split(xml)[-1].split('.')[0]
        if not os.path.isfile(xml.replace('.tif', '.xml')):
            f.write(f'{xml}\n')
            count += 1
    print(f'NORMAL FILES : {count}')
            
print('Done!')
