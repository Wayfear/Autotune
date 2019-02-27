
# Autotune

-----
## Setting
In this part, you need to prepare two computers and some GoPro to build up Autotune System.
### Front-end Config

#### Software Preparation
install opencv and goprocam on Computer1

```
conda config --add channels conda-forge
conda install opencv
conda update --all
pip install goprocam 
```

#### Hardware Preparatiom
* Cameras
    We used GoPro Hero 4 in real world experiments. If cameras is putted in place where is closed to observation objects, we recommend you to set cameras with 720P, 120FPS.  If cameras is putted in place where far away from observation objects, we recommend you to set cameras with 1080P, 30FPS.
* Wifi Sniffer
  You need to use another Computer(Computer2) installed Ubuntu System and T-Shark. Then you can run channel hop scripts in this computer.

## Collecting Data
In this part, we will collect video and wirless data, them assign them to each small session.
### Video
Run following commend in Computer1
```
python
```
### Wireless
Run following commend in Computer2, then this computer will listen packets from other devices.
```
tshark
```

## Analysis Data
In this part, we will capture faces from videos, assign small sessions to final sessions and run Autotune Algrithm.
### Capturing Face From Video

### Assign Session

### Autotune Algrithm
<!-- ## Raspberry Pi 3 (RP3) Config

https://stackoverflow.com/questions/39371772/how-to-install-anaconda-on-raspberry-pi-3-model-b

```
sudo chown -R ubuntu /home/ubuntu/anaconda3 
sudo chmod -R +x /home/ubuntu/anaconda3
```

Back up the native py27
```
mv /usr/bin/python2.7 /usr/bin/python2.7_back
```

Install opencv on RP3
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.1.0/modules \
    -D BUILD_EXAMPLES=ON \
    -DENABLE_PRECOMPILED_HEADERS=OFF ..


mv cv2.cpython-35m-arm-linux-gnueabihf.so

cd /usr/local/lib/python3.4/site-packages/

sudo mv cv2.cpython-34m.so cv2.so

cd ~/.virtualenvs/cv/lib/python3.4/site-packages/

ln -s /usr/local/lib/python3.4/site-packages/cv2.so cv2.so
``` -->




