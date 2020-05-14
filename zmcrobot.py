#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apl 1 2020

@author: ZHONGMC

zmcrobot.py

"""


import os
import json
import time
from threading import Thread
import multiprocessing
import serial
import threading


class ZMCRobot:
    def __init__(self, port="/dev/ttyACM0", baud=115200, timeout=10.0, maxw=1.5, maxv=0.2):
        """ Initialize node, connect to bus, attempt to negotiate topics. """
        self.lock = threading.RLock()
        self.timeout = timeout
        self.synced = False
        self.maxw = maxw
        self.maxv = maxv  
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.voltage = 0.0
        self.w = 0
        self.throttle = 0.0
        self.angle = 0.0
        self.serial = None
        self.setV = 0.0
        self.setW = 0.0
        self._shutdown = False

        try:
            print('Try to open Serial: %s at %d ' % ( port, baud))
            self.serial = serial.Serial(port, baud, timeout=self.timeout*0.5)
        except Exception as e:
            print("Error opening serial:",  e)
            #raise SystemExit

        if self.serial == None:
            print('failded to open serial:  ', port)
        else:
            self.readThread = threading.Thread( target=self.serialReadThread )
            self.readThread.start()
            self.serial.timeout = self.timeout*0.5  # Edit the port timeout
            time.sleep(0.1)           # Wait for ready (patch for Uno)

    def serialReadThread(self):
        print("Start reading thead of serial...")
        data = bytearray ()
        inBinaryPkg = False
        cnt = 0
        binaryPkgLen = -1

        while True:
            if self._shutdown == True:
                print('required to exit ...')
                break
            byteData = self.serial.read()
            if len(byteData) <= 0 :
                continue

            data.append( byteData[0] )

            if inBinaryPkg == True:
                if( binaryPkgLen == -1 ) :
                    binaryPkgLen = byteData[0]
                else :
                    cnt += 1
                    if( cnt >= binaryPkgLen ):
                        self.binaryPackageProcess( data )
                        cnt = 0
                        binaryPkgLen = -1
                        inBinaryPkg = False
                        data.clear()
                continue

            if (byteData[0] & 0xff) > 0x7f : #binary package
                data.clear()
                binaryPkgLen = -1
                inBinaryPkg = True
                continue

            if ( byteData[0] == '\r' or byteData[0] == '\n' or byteData[0] == ';'):
                try:
                    lineStr = data.decode('utf-8')
                    print( lineStr )
                    data.clear()
                except exception as e:
                    print('serial data err: ', e )
                    continue
                if lineStr.find('READY') != -1:
                    print('Robot connected.')
                    self.sendCmd('\ncr;\n' )
                elif lineStr.find('IR') == 0 or lineStr.find('IM') == 0 or lineStr.find('RD') == 0 or lineStr.find('CM') == 0:
                    continue
                else:
                    print('info:', lineStr )
        print('all done! Let s go...')                

    # def serialReadThread(self):
    #     print("\nStart reading thead of serial...")
    #     while True :
    #         if self._shutdown == True:
    #             print('required to end ... ')
    #             break

    #         lineReaded = self.serial.readline()
    #         if len(lineReaded) < 2:
    #             continue

    #         try:
    #             lineStr = lineReaded.decode('utf-8')
    #         except Exception as e:
    #             print('serial data err: ', e )
    #             continue

    #         if lineStr.find('READY') == 0:
    #             print('Robot connected.')
    #             self.sendCmd('\ncr;\n' )
    #             continue
    #         elif lineStr.find('IR') == 0 or lineStr.find('IM') == 0 or lineStr.find('RD') == 0 or lineStr.find('CM') == 0:
    #             continue
    #         elif lineStr.find('RP') == 0:
    #             self.robotPosProcess( lineStr )
    #             continue
    #         else:
    #             print('info:', lineStr )
    #     pass

    def print_position( self ):
        print('x:%.3f,y:%.3f; theta:%.3f;\n v:%.3f, voltage: %.2f' % (self.x, self.y, self.theta, self.v, self.voltage) )
        pass

    def sendCmd(self, data ):

        if self.serial == None:
            return

        self.lock.acquire()
        try:
            self.serial.write( data.encode('utf-8') )
        finally:
            self.lock.release()
        pass

    def binaryPackageProcess(self, data):
        if data[0] == 0xa1 or data[0] == 0xa0:
            self.x = (data[2] & 0xff + data[3]&0xff *256) / 1000.0
            self.y = (data[4] & 0xff + data[5]&0xff *256) / 1000.0
            self.theta = (data[6] & 0xff + data[7]&0xff *256) / 1000.0
            self.v = (data[8] & 0xff ) / 100.0
            self.voltage = (data[9] & 0xff) / 10.0
        pass

    def robotPosProcess( self, posInfo ): 
        #posInfo RBx,y,theta,w,v;
        intStrs = posInfo[2:].split(',')
        try:
            self.x = int(intStrs[0])/10000.0
            self.y = int(intStrs[1])/10000.0
            self.theta = int(intStrs[2])/10000.0
            self.w = int(intStrs[3])/10000.0
            self.v = int(intStrs[4])/10000.0
        except Exception:
            print('RP data err:', posInfo)
        pass

    def drive_car(self, v, w):
        setV = v
        setW = w
        if abs( setV ) < 0.01:
            setV = 0.0
        if abs(setW) < 0.01:
           setW = 0.0
        if setV == self.setV and setW == self.setW:
            return
        self.setV = setV
        self.setW = setW    
        cmdStr = 'sd%.3f,%.3f;\n' % (self.setV, self.setW)
        #cmdStr = 'sd' + str(self.setV) + ',' + str(self.setW) + ';\n'
        self.sendCmd( cmdStr )
        print( cmdStr )
        pass

# donkeyCar api
    def update(self):
        pass

    def run(self, throttle, angle):
        self.throttle = throttle
        self.angle = angle

        setV = throttle * self.maxv
        setW = angle * self.maxw

        if abs(setV) < 0.01:
            setV = 0.0
        if abs(setW) < 0.01:
           setW = 0.0


        if setV == self.setV and setW == self.setW:
            return self.x,self.y,self.theta, self.w, self.v

        self.setV = setV
        self.setW = setW    
        cmdStr = 'sd%.3f,%.3f;\n' % (self.setV, self.setW)
        #cmdStr = 'sd' + str(self.setV) + ',' + str(self.setW) + ';\n'
        self.sendCmd( cmdStr )
        print('ud:%.3f,%.3f' % (throttle, angle))
        #print('drv:', self.setV, self.setW)
        print( cmdStr )
        return self.x,self.y,self.theta, self.w, self.v
        pass

    def run_threaded(self):
        return self.x,self.y,self.theta, self.w, self.v
        pass

    def shutdown(self):
        self._shutdown = True
        
