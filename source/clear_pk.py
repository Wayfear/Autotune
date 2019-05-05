import tools
import os
import yaml
from shutil import rmtree
from os.path import join


project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

thres = cfg['pre_process']['wifi_threshold']
middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'])
middle_pk_path = tools.get_meeting_and_path_list(middle_data_path, r'.+\.pk\d+$')

for k in middle_pk_path:
    for path in middle_pk_path[k]:
        os.remove(path)
