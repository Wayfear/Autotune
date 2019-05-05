
import pandas as pd
import os
from os.path import join
import yaml

project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

mac_address = cfg['mac_name']
macs = list(mac_address.keys())

################
# listen[x] file
################
with open('listen_desktop.sh', 'w') as f:
    f.write('sudo airmon-ng stop mon0\n')
    f.write('sleep 8\n')
    # f.write('sudo airmon-ng check kill\n')
    f.write('sudo airmon-ng start wlan0\n')
    f.write('sudo service network-manager stop\n')
    f.write("now=$(date '+%Y_%m_%d_%H_%M_%S')\n")
    f.write('touch wifi_data/desktop_$now.txt\n')
    f.write('chmod o=rw wifi_data/desktop_$now.txt\n')
    f.write('sleep 2\n')
    f.write('sudo tshark -i mon0 -f "ether src %s'%macs[0])
    for i in range(1, len(macs)):
        f.write(' or ether src %s'%macs[i])
    f.write('" -T fields -e frame.time -e wlan.sa -e radiotap.dbm_antsignal > wifi_data/desktop_$now.txt')

with open('listen.sh', 'w') as f:
    f.write('sudo airmon-ng stop wlan1mon\n')
    f.write('sleep 8\n')
    f.write('sudo airmon-ng start wlan1\n')
    f.write("now=$(date '+%Y_%m_%d_%H_%M_%S')\n")
    f.write('touch /home/hongkaiw/video_scan/wifi_data/$now.txt\n')
    f.write('chmod o=rw /home/hongkaiw/video_scan/wifi_data/$now.txt\n')
    f.write('sleep 2\n')
    f.write('sudo tshark -i wlan1mon -f "ether src %s'%macs[0])
    for i in range(1, len(macs)):
        f.write(' or ether src %s'%macs[i])
    f.write('" -T fields -e frame.time -e wlan.sa -e radiotap.dbm_antsignal > /home/hongkaiw/video_scan/wifi_data/$now.txt')

with open('all_listen.sh', 'w') as f:
    f.write('sudo airmon-ng stop mon0\n')
    f.write('sleep 8\n')
    f.write('sudo airmon-ng start wlan0\n')
    f.write('sudo service network-manager stop\n')
    f.write("now=$(date '+%Y_%m_%d_%H_%M_%S')\n")
    f.write('touch wifi_data/all_$now.txt\n')
    f.write('chmod o=rw wifi_data/all_$now.txt\n')
    f.write('sleep 2\n')
    f.write('sudo tshark -i mon0 -T fields -e frame.time -e wlan.sa -e radiotap.dbm_antsignal > wifi_data/$now.txt')




################
# WiFi file
################

with open('wifi_mod_desktop.sh', 'w') as f:
    f.write('sudo airmon-ng stop mon0\n')
    f.write('sudo service network-manager start')


with open('wifi_mod.sh', 'w') as f:
    f.write('sudo airmon-ng stop wlan1mon')

################
# chanhop file
################
with open('chanhop_desktop.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('IFACE="mon0"\n')
    f.write('IEEE80211bg="1 6 11"\n')
    f.write('IEEE80211bg_intl="$IEEE80211b 12 13 14"\n')
    f.write('IEEE80211a="40 64 100 132"\n')
    f.write('IEEE80211bga="$IEEE80211bg $IEEE80211a"\n')
    f.write('IEEE80211bga_intl="$IEEE80211bg_intl $IEEE80211a"\n')
    f.write('while true ; do\n')
    f.write('for CHAN in $IEEE80211bga ; do\n')
    f.write('echo "Switching to channel $CHAN"\n')
    f.write('iwconfig $IFACE channel $CHAN\n')
    f.write('sleep 1\n')
    f.write('done\n')
    f.write('done')

with open('chanhop.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('IFACE="wlan1mon"\n')
    f.write('IEEE80211bg="1 6 11"\n')
    f.write('IEEE80211bg_intl="$IEEE80211b 12 13 14"\n')
    f.write('IEEE80211a="40 64 100 132"\n')
    f.write('IEEE80211bga="$IEEE80211bg $IEEE80211a"\n')
    f.write('IEEE80211bga_intl="$IEEE80211bg_intl $IEEE80211a"\n')
    f.write('while true ; do\n')
    f.write('for CHAN in $IEEE80211bga ; do\n')
    f.write('echo "Switching to channel $CHAN"\n')
    f.write('iwconfig $IFACE channel $CHAN\n')
    f.write('sleep 1\n')
    f.write('done\n')
    f.write('done')

