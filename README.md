# AutoTune

## Introduction

This is an implementation of paper [Autonomous Learning for Face Recognition in the Wildvia Ambient Wireless Cues]()

## Install

### Environment

* OpenCV
* tensorflow 1.4
* Tshark
* [goprocam](https://pypi.org/project/goprocam/)

## How to run

### Preparation

#### Wifi data caputre

1. Set your router to make sure the router to have fixed channels
2. Prepare a computer running ubuntu 14.04
3. Add mac address which you want to listen to `config.yaml` file. Example like that:

   ```yaml
   mac_name:
    'device:mac:address': device's name
    ...
   ```

4. Run `create_listen_bash.py` to create bash scripts.
   * Firstly, run `listen.sh` to create interface for listening
   * Secondly, run `chanhop.sh` to listen the fixed channel set in step 1, the data file will be saved in `{root_folder}/wifi_data/`.
   * If you want to stop listening process, kill the two bash above, then run `wifi_mod.sh` to reset the network interface card.
  
    Tips: Due to the differences of network interface cards, you may need different commends to start the listening process. Although `create_listen_bash.py` script will create two groups of bash script for different devices, it is very likely that none of them can work on your devices. You may need to try the right commend in your decice. The network interface card we use is [TP-Link TL-WN726N](https://www.tp-link.com.cn/product_494.html)

#### Video Data caputre

1. To begin with, you will prapare some `GoPro Hero 4`, setting these device's video quality to 120fps and 720p.
2. Set these devices to Wi-Fi mode and then connect your computer to the GoPro's Wi-Fi.
3. Make sure your computer installed [goprocam](https://pypi.org/project/goprocam/) and run `live_stream.py` to capture video.

### Data Preprocess

#### Wi-Fi Data

* Run `analysis_wifi.py` to preprocess the Wi-Fi data. the result will be saved in `{root_folder}/origin_data/`

#### Video Data

1. Run `download_video_from_gopro.py` to download these videos captured by gopro, these videos will be saved in `{root_folder}/video/` and be named by those videos' capture time.
2. Run `assign_video_by_shot_time.py` to group videos based on capture time, these videos will be moved to `{root_folder}/origin_data/`.

#### Preprocess

1. Run `source/pre_process.py` to convert videos to aligned face images and the result will be saved in `{root_folder}/middle_data/time/`.

2. Run `source/get_meeting_data.py` to assgin data based on session setting.

#### Label Data

During tuning process, Autotune framework will evaluate the framework performance iteratively.

* Runing `source/feature_extraction_classifier.py`
  
* After running, the structure of result folder is showing below. You need to correct the classifier error in the `classifier` folder manually.
  
    ```bash
    middle_data/
    ├── 03-22-11-00-00_03-22-14-00-00
    │   ├── classifier
    │   │   ├── 0 # people 0's images
    │   │   │   ├── 03-22-12-20-00_03-22-12-40-00_4787_1079.jpeg
    │   │   │   ├── 03-22-12-40-00_03-22-13-00-00_134_2127.jpeg
    │   │   │   ├── 03-22-12-40-00_03-22-13-00-00_135_2251.jpeg
    │   │   │   ├── 03-22-12-40-00_03-22-13-00-00_136_2079.jpeg
    │   │   │   └── 03-22-12-40-00_03-22-13-00-00_137_2389.jpeg
    │   │   └── 1 # people 1's images
    │   │       ├── 03-22-12-40-00_03-22-13-00-00_139_2337.jpeg
    │   │       ├── 03-22-12-40-00_03-22-13-00-00_140_2317.jpeg
    │   │       ├── 03-22-12-40-00_03-22-13-00-00_141_2300.jpeg
    │   │       ├── 03-22-13-40-00_03-22-14-00-00_4124_7513.jpeg
    │   │       └── 03-22-13-40-00_03-22-14-00-00_4125_6286.jpeg
    │   └── mtcnn # origin data
    │       ├── 03-22-12-20-00_03-22-12-40-00_4787_1079.jpeg
    │       ├── 03-22-12-40-00_03-22-13-00-00_134_2127.jpeg
    │       ├── 03-22-12-40-00_03-22-13-00-00_135_2251.jpeg
    │       ├── 03-22-12-40-00_03-22-13-00-00_136_2079.jpeg
    │       ├── 03-22-12-40-00_03-22-13-00-00_137_2389.jpeg
    │       ├── 03-22-12-40-00_03-22-13-00-00_139_2337.jpeg
    │       ├── 03-22-12-40-00_03-22-13-00-00_140_2317.jpeg
    │       ├── 03-22-12-40-00_03-22-13-00-00_141_2300.jpeg
    │       ├── 03-22-13-40-00_03-22-14-00-00_4124_7513.jpeg
    │       └── 03-22-13-40-00_03-22-14-00-00_4125_6286.jpeg
    └── 03-22-14-00-00_03-22-17-00-00
        ...
    ```

### AutoTune Hard

```bash
cd source
./auto_em_hard.sh
```

### AutoTune Soft

```bash
cd source
./auto_em.sh
```