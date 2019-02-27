
# video_scan

-----
## Front-end Config

install opencv on laptop

```
conda config --add channels conda-forge
conda install opencv
conda update --all
pip install goprocam 
```
* Cameras@Oxford
    * gopro_front: 720, 240fps, N(arrow)
    * lalala: 1080, 120fps, N 

## Raspberry Pi 3 (RP3) Config

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
```




