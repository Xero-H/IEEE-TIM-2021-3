import threading
import time
import torch
import torch.utils.data as Data
import numpy as np

import smbus
from time import sleep
from model.baseline_wisdm import *
import tkinter as tk

# select the correct i2c bus for this revision of Raspberry Pi
revision = ([l[12:-1] for l in open('/proc/cpuinfo', 'r').readlines() if l[:8] == "Revision"] + ['0000'])[0]
bus = smbus.SMBus(1 if int(revision, 16) >= 4 else 0)

# ADXL345 constants
EARTH_GRAVITY_MS2 = 9.80665
SCALE_MULTIPLIER = 0.004

DATA_FORMAT = 0x31
BW_RATE = 0x2C
POWER_CTL = 0x2D

BW_RATE_1600HZ = 0x0F
BW_RATE_800HZ = 0x0E
BW_RATE_400HZ = 0x0D
BW_RATE_200HZ = 0x0C
BW_RATE_100HZ = 0x0B
BW_RATE_50HZ = 0x0A
BW_RATE_25HZ = 0x09

RANGE_2G = 0x00
RANGE_4G = 0x01
RANGE_8G = 0x02
RANGE_16G = 0x03

MEASURE = 0x08
AXES_DATA = 0x32


class ADXL345:
    address = None

    def __init__(self, address=0x53):
        self.address = address
        self.setBandwidthRate(BW_RATE_100HZ)
        self.setRange(RANGE_2G)
        self.enableMeasurement()

    def enableMeasurement(self):
        bus.write_byte_data(self.address, POWER_CTL, MEASURE)

    def setBandwidthRate(self, rate_flag):
        bus.write_byte_data(self.address, BW_RATE, rate_flag)

    # set the measurement range for 10-bit readings
    def setRange(self, range_flag):
        value = bus.read_byte_data(self.address, DATA_FORMAT)

        value &= ~0x0F;
        value |= range_flag;
        value |= 0x08;

        bus.write_byte_data(self.address, DATA_FORMAT, value)

    # returns the current reading from the sensor for each axis
    #
    # parameter gforce:
    #    False (default): result is returned in m/s^2
    #    True           : result is returned in gs
    def getAxes(self, gforce=False):
        bytes = bus.read_i2c_block_data(self.address, AXES_DATA, 6)

        x = bytes[0] | (bytes[1] << 8)
        if (x & (1 << 16 - 1)):
            x = x - (1 << 16)

        y = bytes[2] | (bytes[3] << 8)
        if (y & (1 << 16 - 1)):
            y = y - (1 << 16)

        z = bytes[4] | (bytes[5] << 8)
        if (z & (1 << 16 - 1)):
            z = z - (1 << 16)

        x = x * SCALE_MULTIPLIER
        y = y * SCALE_MULTIPLIER
        z = z * SCALE_MULTIPLIER

        if gforce == False:
            x = x * EARTH_GRAVITY_MS2
            y = y * EARTH_GRAVITY_MS2
            z = z * EARTH_GRAVITY_MS2

        # x = round(x, 4)
        # y = round(y, 4)
        # z = round(z, 4)

        return {"x": x, "y": y, "z": z}


adxl345_data = [0, 0, 0]
adxl345_sum_data = []




def collect_data():
    adxl345 = ADXL345()
    while True:
        axes = adxl345.getAxes(False)
        adxl345_data[0] = axes['x']
        adxl345_data[1] = axes['y']
        adxl345_data[2] = axes['z']
        adxl345_sum_data.append(adxl345_data)
        if len(adxl345_sum_data) > 200:
            time.sleep(0.02)
            del(adxl345_sum_data[:-200])


def gui():
    model = torch.load('./model_save/wisdm/net0.9818016378525932_129.pth', map_location='cpu')
    model = model.module.to(torch.device("cpu"))

    def changeImage(preds):
        if preds == 0:
            label_img.configure(image=img_gif0)
            label_img.after(1, changeImage, preds)
        elif preds == 1:
            label_img.configure(image=img_gif1)
            label_img.after(1, changeImage, preds)
        elif preds == 2:
            label_img.configure(image=img_gif2)
            label_img.after(1, changeImage, preds)
        elif preds == 3:
            label_img.configure(image=img_gif3)
            label_img.after(1, changeImage, preds)
        elif preds == 4:
            label_img.configure(image=img_gif4)
            label_img.after(1, changeImage, preds)
        elif preds == 5:
            label_img.configure(image=img_gif5)
            label_img.after(1, changeImage, preds)

    while True:
        top = tk.Tk()

        top.title("HAR_demo")
        width = 260
        height = 380
        top.geometry(f'{width}x{height}')

        img_gif = tk.PhotoImage(file='./动作/问号.gif')
        img_gif0 = tk.PhotoImage(file='./动作/走.gif')
        img_gif1 = tk.PhotoImage(file='./动作/上楼.gif')
        img_gif2 = tk.PhotoImage(file='./动作/下楼.gif')
        img_gif3 = tk.PhotoImage(file='./动作/坐.gif')
        img_gif4 = tk.PhotoImage(file='./动作/站立.gif')
        img_gif5 = tk.PhotoImage(file='./动作/躺.gif')

        label_img = tk.Label(top, image=img_gif)
        label_img.place(x=30, y=160)  # 30  120

        sensor_data = np.array(adxl345_sum_data[-200:])
        data_x = sensor_data.reshape(-1, 1, 200, 3)  # (N, C, H, W) (7352, 1, 128, 9)
        inputs = torch.tensor(data_x).float()

        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            # preds_prob = torch.nn.functional.softmax(outputs.data, dim=1)  # ####################
            _, preds = torch.max(outputs, 1)
            print(preds)




        label_img.after(1, changeImage, preds)
        top.update_idletasks()

        top.mainloop()


thread_1 = threading.Thread(target=collect_data, name="T1")
thread_2 = threading.Thread(target=gui, name="T2")

thread_1.start()
thread_2.start()

