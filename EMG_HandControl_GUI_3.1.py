# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:57:07 2022

@author: mppen

This is the UI code for the EMG Hand Control code. It operates on the PyQt5, 
Mindrove BrainFlow, and nidaqmx libraries.
"""

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.11.3

import nidaqmx
import logging
import webbrowser
import traceback
import beepy as bp
import numpy as np
import time
from datetime import datetime
import os
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import (
    QObject,
    QThread,
    pyqtSignal,
    QRect,
    pyqtSlot,
    Qt
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMenu,
    QAction,
    QMessageBox,
    QWidget,
    QPushButton,
    QApplication,
    QListWidget,
    QLabel,
    QPlainTextEdit,
    QStatusBar,
    QMainWindow
)
from mindrove_brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds       

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


global count, train_data, prompt, model


model = None
prompt = [0,2,0,1,-1]
count = -1
train_reps = len(prompt) - 1
train_data = np.array([], dtype=np.int64).reshape(0,9)
    
# use synthetic board for demo
params = BrainFlowInputParams()
board = BoardShim(BoardIds.MINDROVE_WIFI_BOARD.value, params)
'''
task_write = nidaqmx.task.Task("volOUT")
task_write.ao_channels.add_ao_voltage_chan("DevHoH/ao0", 'TIM', 0,3.3)
task_write.ao_channels.add_ao_voltage_chan("DevHoH/ao1", 'RL', 0,3.3)

task_read = nidaqmx.task.Task("volIN")
task_read.ai_channels.add_ai_voltage_chan("DevHoH/ai0")
task_read.ai_channels.add_ai_voltage_chan("DevHoH/ai1")
'''
pos = {
       "rest":[1.65,1.65],
       "open":[3.3,3.3],
       "close":[0.0,0.0],
       "tripod":[1.65,3.3],
       "tripod_open":[3.3,1.65]
       }

pos_list = tuple(pos.values())


pos_rev = {
       "rest":[1.65,1.65],
       "reverse open":[0.0,0.0],
       "reverse close":[3.3,3.3],
       "reverse tripod":[1.65,0.0],
       "reverse tripod_open":[0.0,1.65]
       }

pos_rev_list = tuple(pos_rev.values())

imgs = (
        "imgs/rest.jpg",
        "imgs/open.jpg",
        "imgs/close.jpg",
        "imgs/tripod.jpg",
        "imgs/tripod_open.jpg",
        "imgs/blank.jpg"
        )

#Bandpass filter functions
def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def rolling_rms(x, N):
    return (pd.DataFrame(abs(x)**2).rolling(N).mean()) **0.5

def window_rms(a, window_size):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))


logging.basicConfig(format="%(message)s", level=logging.INFO)
            
class timeWorker(QObject):
    finished = pyqtSignal()
    updateTime = pyqtSignal(int)
    
    def run(self):
        t=5
        while t:
            self.updateTime.emit(t)
            time.sleep(1)
            t-=1
        self.finished.emit()

class collectWorker(QObject):
    
    finished = pyqtSignal()
    updateImg = pyqtSignal()
    updateData = pyqtSignal()

    def stream(self):
        global train_data
        
        timenow = int(datetime.now().strftime('%Y%m%d%H%M%S'))
        for i in range(train_reps):
            self.updateImg.emit()
            self.updateData.emit()
            logging.info("Collcting data")
            k = prompt[i]
            board.start_stream()
            time.sleep(4)
            board.stop_stream()
            data = board.get_board_data()
            
            # Processing Data
            data[8,:]=int(k)
            data = np.transpose(data)
            dataDF = pd.DataFrame(data[:,0:9],columns =['CH0','CH1','CH2','CH3','CH4','CH5','CH6','CH7','Pose'])
            data = dataDF.reset_index(drop=True)
            original = pd.DataFrame(columns = ['CH0','CH1','CH2','CH3','CH4','CH5','CH6','CH7']) 
            filtered = pd.DataFrame(columns = ['CH0','CH1','CH2','CH3','CH4','CH5','CH6','CH7'])
            smoothed = pd.DataFrame(columns = ['CH0','CH1','CH2','CH3','CH4','CH5','CH6','CH7','Pose'])
            
            fs = 500
            fhc = 200
            flc = 4
            ws = 50
            
            for i in filtered.columns: 
                
                original[i] = data[i]
                filtered[i] = butter_bandpass_filter(data[i], flc, fhc, fs, order=6)
                smoothed[i] = window_rms(filtered[i], ws)
            
            smoothed.Pose = data.Pose 
            train_data = np.concatenate((train_data,smoothed))
        self.updateImg.emit()
        train_data = pd.DataFrame(train_data, columns=['CH0','CH1','CH2','CH3','CH4','CH5','CH6','CH7','Pose'])
        train_data.to_csv(f"EMG_data/emg_data{timenow}.csv")

    def run(self):
        """Long-running task."""
        
        if board.is_prepared():
            self.stream()
        else:
            board.prepare_session()
            self.stream()
        board.release_session()
        self.finished.emit()

class trainWorker(QObject):
    
    finished = pyqtSignal()
    newName = pyqtSignal()
    updateName = pyqtSignal()
    
    def run(self):
        global model
        
        if model == None:
            
            #%%
                            
            time = np.linspace(1, len(train_data.CH1)/500, num=len(train_data.CH1))
            
            Pose = plt.plot(time,train_data.Pose)
            plt.show()
            CH1 = plt.plot(time,train_data.CH1)
            
            #%%from sklearn.metrics import classification_report
            
            X = train_data[["CH0","CH1","CH2","CH3","CH4","CH5","CH6","CH7"]]
            
            def get_one_hot(targets, nb_classes):
                res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
                return res.reshape(list(targets.shape)+[nb_classes])
            
            y=get_one_hot(train_data.Pose.astype(int),3)
            
            X = StandardScaler().fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
            
            #%%
            
            model = Sequential()
            model.add(Dense(2500,activation='relu',input_shape=(8,)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(3, activation="softmax"))
            
            #%%
            model.compile(loss='categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy'])
            model.fit(X_train, y_train,epochs=50, batch_size=2, verbose=1)
            
            #%%
            y_predicted = model.predict(X)
            
            y_predicted[y_predicted[:,:] >0.7] = 1
            y_predicted[y_predicted[:,:] != 1] = 0
            
            pose_predict = y_predicted[:,0]*0 + y_predicted[:,1]*1 + y_predicted[:,2]*2
            pose_train = y[:,0]*0 + y[:,1]*1 +y[:,2]*2
            
            plt.show()
            plt.plot(time, pose_train)
            plt.scatter(time, pose_predict, color = "orange",s = 6)
            plt.legend(["Actual","Estimated"],loc = 'upper left')
            plt.yticks([0,1,2],['Rest','Open','Closed'])
            plt.xlabel("Time [s]")
            k = accuracy_score(pose_predict,  pose_train  )
            title = "Results using Deep Learning" "\n Accuracy: " + str(k*100) + "%"
            plt.title(title)
            plt.show()
            self.newName.emit()
        else:
            #%% 
            
            time = np.linspace(1, len(train_data.CH1)/500, num=len(train_data.CH1))
            Pose = plt.plot(time,train_data.Pose)
            plt.show()
            CH1 = plt.plot(time,train_data.CH1)
            
            #%%from sklearn.metrics import classification_report
            
            X = train_data[["CH0","CH1","CH2","CH3","CH4","CH5","CH6","CH7"]]
            
            def get_one_hot(targets, nb_classes):
                res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
                return res.reshape(list(targets.shape)+[nb_classes])
            
            y=get_one_hot(train_data.Pose.astype(int),3)
            
            X = StandardScaler().fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
            
            #%%
            
            model.compile(loss='categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy'])
            model.fit(X_train, y_train,epochs=50, batch_size=2, verbose=1)
            
            #%%
            y_predicted = model.predict(X)
            
            y_predicted[y_predicted[:,:] >0.7] = 1
            y_predicted[y_predicted[:,:] != 1] = 0
            
            pose_predict = y_predicted[:,0]*0 + y_predicted[:,1]*1 + y_predicted[:,2]*2
            pose_train = y[:,0]*0 + y[:,1]*1 +y[:,2]*2
            
            plt.show()
            plt.plot(time, pose_train)
            plt.scatter(time, pose_predict, color = "orange",s = 6)
            plt.legend(["Actual","Estimated"],loc = 'upper left')
            plt.yticks([0,1,2],['Rest','Open','Tripod Open'])
            plt.xlabel("Time [s]")
            k = accuracy_score(pose_predict,  pose_train  )
            title = "Results using Deep Learning" "\n Accuracy: " + str(k*100) + "%"
            plt.title(title)
            plt.show()
            self.updateName.emit()
        self.finished.emit()

class testWorker(QObject):
    
    finished = pyqtSignal()
    updateImg = pyqtSignal()
    decodedPoses = pyqtSignal(list)
    
    def test(self):
        global train_data
        
        timenow = int(datetime.now().strftime('%Y%m%d%H%M%S'))
        
        task_write.start()
        task_read.start()
        
        poses = list()
        
        fs = 500
        fhc = 200
        flc = 4
        ws = 50
        
        for i in range(train_reps):
            self.updateImg.emit()
            k = prompt[i]
            board.start_stream()
            time.sleep(4)
            board.stop_stream()
            data = board.get_board_data()
            
            # Processing Data
            data[8,:]=int(k)
            data = np.transpose(data)
            dataDF = pd.DataFrame(data[:,0:9],columns =['CH0','CH1','CH2','CH3','CH4','CH5','CH6','CH7','Pose'])
            data = dataDF.reset_index(drop=True)         
            
            original = pd.DataFrame(columns = ['CH0','CH1','CH2','CH3','CH4','CH5','CH6','CH7']) 
            filtered = pd.DataFrame(columns = ['CH0','CH1','CH2','CH3','CH4','CH5','CH6','CH7'])
            smoothed = pd.DataFrame(columns = ['CH0','CH1','CH2','CH3','CH4','CH5','CH6','CH7','Pose'])
            
            for i in filtered.columns: 
                
                original[i] = data[i]
                filtered[i] = butter_bandpass_filter(data[i], flc, fhc, fs, order=6)
                smoothed[i] = window_rms(filtered[i], ws)
            
            smoothed.Pose = data.Pose  
            
            X = smoothed[["CH0","CH1","CH2","CH3","CH4","CH5","CH6","CH7"]]
            X = StandardScaler().fit_transform(X)
            Y = model.predict(X)
            
            Y[Y[:,:] >0.7] = 1
            Y[Y[:,:] != 1] = 0
            pose = np.argmax(np.count_nonzero(Y,axis=0))
            print(pose)
            
            vals = pos_list[pose]
            task_write.write(vals)
            vals = task_read.read()
            tim_val = round(vals[0], 5); rl_val = round(vals[1], 5)
            print(f"TIM: {tim_val}, RL: {rl_val}")
            time.sleep(2)
            
            vals = pos_list[0]
            task_write.write(vals)
            vals = task_read.read()
            tim_val = round(vals[0], 5); rl_val = round(vals[1], 5)
            print(f"TIM: {tim_val}, RL: {rl_val}")
            time.sleep(2)
            
            vals = pos_rev_list[pose]
            task_write.write(vals)
            vals = task_read.read()
            tim_val = round(vals[0], 5); rl_val = round(vals[1], 5)
            print(f"TIM: {tim_val}, RL: {rl_val}")
            time.sleep(2)
            
            vals = pos_list[0]
            task_write.write(vals)
            
            poses.append(pose)
            train_data = np.concatenate((train_data,smoothed))
        self.updateImg.emit()
        self.decodedPoses.emit(poses)
        train_data = pd.DataFrame(train_data, columns=['CH0','CH1','CH2','CH3','CH4','CH5','CH6','CH7','Pose'])
        train_data.to_csv(f"EMG_data/emg_data{timenow}.csv")
        

    
    def run(self):
        if board.is_prepared():
           self.test()
        else:
            board.prepare_session()
            self.test()
        board.release_session()
        task_write.stop()
        #task_read.stop()
        self.finished.emit()
   

class MainWin(QMainWindow):
    
    def __init__(self):
        # Call the Parent constructor
        super().__init__()
        # Set the title of the window
        self.setWindowTitle("EMG Hand Calibrations")
        # Set the geometry of the window
        self.setGeometry(200, 100, 1200, 870)
        
        # Create photo label
        self.photo = QLabel(self)
        self.photo.setGeometry(QRect(0, 0, 900, 800))
        
        # Create push button
        self.collect = QPushButton("COLLECT DATA", self)
        self.collect.setGeometry(QRect(45, 810, 240, 40))
        
        # Create push buttons
        self.train = QPushButton("TRAIN DATA", self)
        self.train.setGeometry(QRect(330, 810, 240, 40))
        
        # Create push buttons
        self.test = QPushButton("TEST DATA", self)
        self.test.setGeometry(QRect(615, 810, 240, 40))
        
        # Display Instructions
        self.instruct = QPlainTextEdit(self)
        self.instruct.setGeometry(QRect(900, 0, 300, 435))
        file=open('EMG_Instructions.txt')
        text = file.read()
        self.instruct.setPlainText(text)
        file.close()
        self.instruct.setReadOnly(True)
        
        # Message Board
        self.message = QPlainTextEdit(self)
        self.message.setGeometry(QRect(900, 435, 300, 435))
        self.message.setReadOnly(True)
        
        # Top display
        self.topdisp = QLabel("", self)
        self.topdisp.setGeometry(QRect(410, 20, 250, 20))
        
        # Connect image and auxiliary functions
        self.collect.pressed.connect(self.collectData)
        
        self.train.pressed.connect(self.trainData)
        
        self.test.pressed.connect(self.testData)
        
        self._createActions()
        self._createMenuBar()
        self._connectActions()
        
        # Display the window
        self.show()
        
        # Initiliaze DAQ hardware
        self._init_DAQ()
    
    def _createActions(self):
        # Creating actions using the second constructor
        self.loadDataAction = QAction("Load &Data...", self)
        self.loadModelAction = QAction("Load &Model...", self)
        self.saveAction = QAction("&Save", self)
        self.exitAction = QAction("&Exit", self)
        self.copyAction = QAction("&Copy", self)
        self.pasteAction = QAction("&Paste", self)
        self.cutAction = QAction("C&ut", self)
        self.documentationAction = QAction("&Documentation", self)
        self.aboutAction = QAction("&About", self)
        self.newPromptAction = QAction("&New Prompt...", self)
        self.randPromptAction = QAction("&Random Prompt", self)
        self.DAQAction = QAction("&NI DAQ", self)
        self.mindroveAction = QAction("&MindRove", self)
        self.testAllAction = QAction("Test &All", self)
        self.uploadAction = QAction("&Upload...", self)
        
    def _createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = menuBar.addMenu("File")
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.loadDataAction)
        fileMenu.addAction(self.loadModelAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.exitAction)
        # Tools menu
        toolMenu = menuBar.addMenu("Tools")
        toolMenu.addAction(self.newPromptAction)
        toolMenu.addAction(self.randPromptAction)
        hardwareMenu = toolMenu.addMenu("Hardware Tests")
        hardwareMenu.addAction(self.DAQAction)
        hardwareMenu.addAction(self.mindroveAction)
        hardwareMenu.addAction(self.testAllAction)
        # Help menu
        helpMenu = menuBar.addMenu("Help")
        helpMenu.addAction(self.documentationAction)
        helpMenu.addAction(self.aboutAction)
        
    
    def _connectActions(self):
        # Connect File actions
        self.loadDataAction.triggered.connect(self.loadData)
        self.loadModelAction.triggered.connect(self.loadModel)
        self.saveAction.triggered.connect(self.saveFile)
        self.newPromptAction.triggered.connect(self.newPrompt)
        self.randPromptAction.triggered.connect(self.randPrompt)
        self.DAQAction.triggered.connect(self.testDAQ)
        self.mindroveAction.triggered.connect(self.testMindrove)
        self.testAllAction.triggered.connect(self.testAll)
        self.exitAction.triggered.connect(self.close)

        # Connect Help actions
        self.documentationAction.triggered.connect(self.documentation)
        self.aboutAction.triggered.connect(self.about)

    def loadData(self):
        global train_data
        
        fname, selFilter = QFileDialog.getOpenFileName(self, 'Open file', 
                                            'c:\\',"CSV files(*.csv);;XML files(*.xml);;Text files (*.txt)")
        
        train_data = pd.read_csv(fname)
        self.message.setPlainText(f"{fname} successfully uploaded!")
    
    def loadModel(self):
        global model
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        model = keras.models.load_model(folder)
        self.message.setPlainText(f"{folder} successfully uploaded!")
        
    
    def saveFile(self):
        print(len(train_data))
        print(prompt)
        # Logic for saving a file goes here...
        self.topdisp.setText("<b>File > Save...</b> clicked")
    
    def newPrompt(self):
        global prompt, train_reps
        while True:
            prompt, done1 = QInputDialog.getText(
                self, 'Input Dialogue', 'Enter new Prompt:')
            prompt = prompt.split()
            for i in range(len(prompt)):
                if prompt[i].isdigit():
                    prompt[i] = int(prompt[i])

            if all([isinstance(item, int) for item in prompt]):
                if all([item >= 0 for item in prompt]) and all([item <= 4 for item in prompt]):
                    self.topdisp.setText(f"New prompt of {str(prompt)} created!")
                    prompt.append(-1)
                    train_reps = len(prompt) - 1
                    break
                else:
                    self.message.setPlain("Integers should be between 0 and 4")
                    continue
            else:
                self.message.setPlainText("Create input as integers seperated by space")
                continue
    
    def randPrompt(self):
        global prompt, train_reps
        length = np.random.randint(4, high=9)
        prompt = np.random.randint(0,high=5,size=(1,length))
        prompt = prompt.tolist()
        prompt = prompt[0]
        self.topdisp.setText(f"New prompt of {str(prompt)} created!")
        prompt.append(-1)
        train_reps = len(prompt) - 1
    
    def testDAQ(self):
        # Logic for saving a file goes here...
        self.topdisp.setText("<b>Tool > Hardware Tests > NI DAQ</b> clicked")
        
    def testMindrove(self):
        # Logic for saving a file goes here...
        self.topdisp.setText("<b>Tool > Hardware Tests > MindRove</b> clicked")
            
    def testAll(self):
        # Logic for saving a file goes here...
        self.topdisp.setText("<b>Tool > Hardware Tests > Test All</b> clicked")

    def documentation(self):
        webbrowser.open_new(r'HoH EMG Hand Documentation Version 2.0.pdf')

    def about(self):
        # Logic for showing an about dialog content goes here...
        self.topdisp.setText("<b>Help > About...</b> clicked")
    
    def closeEvent(self, event):
        if board.is_prepared():
            close = QMessageBox.question(self,
                                               "QUIT",
                                               "ARE YOU SURE? Quitting will interrupt data collection",
                                               QMessageBox.Yes | QMessageBox.No)
            if close == QMessageBox.Yes: # add boolean statement for threads
                board.release_session()
                task_write.close()
                task_read.close()
                event.accept()
            elif close == QMessageBox.No:
                event.ignore()
        else:
            task_write.close()
            task_read.close()
            event.accept()
   
    def _init_DAQ(self):
        global task_write, task_read
        try:
            task_write = nidaqmx.task.Task("volOUT")
            task_write.ao_channels.add_ao_voltage_chan("DevHoH/ao0", 'TIM', 0,3.3)
            task_write.ao_channels.add_ao_voltage_chan("DevHoH/ao1", 'RL', 0,3.3)

            task_read = nidaqmx.task.Task("volIN")
            task_read.ai_channels.add_ai_voltage_chan("DevHoH/ai0")
            task_read.ai_channels.add_ai_voltage_chan("DevHoH/ai1")
        except:
            self.message.setPlainText("DaqError: Device cannot be accessed. Ensure the Device is in the system and powered.")
            
            #self._init_DAQ()
    #def data_progress(self):
        #self.topdisp.setText(f"COLLECTING DATA FROM POSE {prompt[count]}")
   
    # Result Functions
   
    def countdown_thread_complete(self):
        logging.info("COUNTDOWN THREAD COMPLETE!")
        self.topdisp.setText("BEGIN TESTING")
    
    def collect_thread_complete(self):
        logging.info("Collected All Data")
        logging.info("DATA THREAD COMPLETE!")
        self.topdisp.setText("COLLECTED AND PROCESSED ALL DATA")
        
    def test_thread_complete(self):
        logging.info("TEST THREAD COMPLETE!")
        self.topdisp.setText("TESTING FINISHED")
    
    def train_thread_complete(self):
        logging.info("TRAIN THREAD COMPLETE!")
        self.topdisp.setText("TRAINING FINISHED")
    
    # AUXILARY FUNCTIONS: Functions receive a signal from workers                

    # Progress Functions: updates progress for each thread
    def time_progress(self, time):
        self.topdisp.setText(f"BEGINNING IN: {time}") 

    def changeImg(self):
        global count
        if count == len(prompt)-2:
            count = -1
            self.photo.setPixmap((QtGui.QPixmap(imgs[prompt[count]])))

            bp.beep(sound=5)
        else:
            count += 1
            self.photo.setPixmap((QtGui.QPixmap(imgs[prompt[count]])))
            bp.beep(sound=1)
            
    def saveNewModel(self):
        fname, done1 = QInputDialog.getText(
            self, 'Create Model', 'Enter name for new model')
        model.save(f"Models/{fname}")
        self.topdisp.setText(f"{fname} successfully created and trained!")
    
    def saveUpdatedModel(self):
        fname, done1 = QInputDialog.getText(
            self, 'Update Model', 'Enter name for updated model')
        model.save(f"Models/{fname}")
        self.topdisp.setText(f"{fname} successfully retrained!")
    
    # Call Tasks: Each task that is assigned a thread
        
    def collectData(self):
        # Create QThread objects
        self.timeThread = QThread()
        self.collectThread = QThread()
       
        # Create worker objects
        self.collectWorker = collectWorker()
        self.timeWorker = timeWorker()
        
        # Move workers to the threads
        self.collectWorker.moveToThread(self.collectThread)
        self.timeWorker.moveToThread(self.timeThread)
        # Connect signals and slots
        self.collectThread.started.connect(self.collectWorker.run)
        self.collectWorker.finished.connect(self.collect_thread_complete)
        self.collectWorker.finished.connect(self.collectThread.quit)
        self.collectWorker.finished.connect(self.collectWorker.deleteLater)
        self.collectThread.finished.connect(self.collectThread.deleteLater)
        self.collectWorker.updateImg.connect(self.changeImg)
        
        self.timeThread.started.connect(self.timeWorker.run)
        self.timeWorker.finished.connect(self.countdown_thread_complete)
        self.timeWorker.finished.connect(self.collectThread.start)
        self.timeWorker.finished.connect(self.timeThread.quit)
        self.timeWorker.finished.connect(self.timeWorker.deleteLater)
        self.timeThread.finished.connect(self.timeThread.deleteLater)
        self.timeWorker.updateTime.connect(self.time_progress)
        # Start the thread
        
        self.timeThread.start()
        self.collect.setEnabled(False)
        self.train.setEnabled(False)
        self.test.setEnabled(False)
        
        # Reset tasks
        self.collectThread.finished.connect(
            lambda: self.collect.setEnabled(True)
            )
        self.collectThread.finished.connect(
            lambda: self.train.setEnabled(True)
            )
        self.collectThread.finished.connect(
            lambda: self.test.setEnabled(True)
            )

    def testData(self):
        # Create QThread objects
        self.timeThread = QThread()
        self.testThread = QThread()
       
        # Create worker objects
        self.testWorker = testWorker()
        self.timeWorker = timeWorker()
        
        # Move workers to the threads
        self.testWorker.moveToThread(self.testThread)
        self.timeWorker.moveToThread(self.timeThread)
        # Connect signals and slots
        self.testThread.started.connect(self.testWorker.run)
        self.testWorker.finished.connect(self.test_thread_complete)
        self.testWorker.finished.connect(self.testThread.quit)
        self.testWorker.finished.connect(self.testWorker.deleteLater)
        self.testThread.finished.connect(self.testThread.deleteLater)
        self.testWorker.updateImg.connect(self.changeImg)
        
        self.timeThread.started.connect(self.timeWorker.run)
        self.timeWorker.finished.connect(self.countdown_thread_complete)
        self.timeWorker.finished.connect(self.testThread.start)
        self.timeWorker.finished.connect(self.timeThread.quit)
        self.timeWorker.finished.connect(self.timeWorker.deleteLater)
        self.timeThread.finished.connect(self.timeThread.deleteLater)
        self.timeWorker.updateTime.connect(self.time_progress)
        # Start the thread
        
        if model == None:
            self.message.setPlainText("CANNOT TEST NON-EXISTENT MODEL")
        else:
            self.timeThread.start()
            self.collect.setEnabled(False)
            self.train.setEnabled(False)
            self.test.setEnabled(False)
        
            # Reset tasks
            self.testThread.finished.connect(
                lambda: self.collect.setEnabled(True)
                )
            self.testThread.finished.connect(
                lambda: self.train.setEnabled(True)
                )
            self.testThread.finished.connect(
                lambda: self.test.setEnabled(True)
                )

    
    def trainData(self):
        # Create QThread objects
        self.trainThread = QThread()
       
        # Create worker objects
        self.trainWorker = trainWorker()
        
        # Move workers to the threads
        self.trainWorker.moveToThread(self.trainThread)

        # Connect signals and slots
        self.trainThread.started.connect(self.trainWorker.run)
        self.trainWorker.finished.connect(self.train_thread_complete)
        self.trainWorker.finished.connect(self.trainThread.quit)
        self.trainWorker.finished.connect(self.trainWorker.deleteLater)
        self.trainThread.finished.connect(self.trainThread.deleteLater)
        self.trainWorker.newName.connect(self.saveNewModel)
        self.trainWorker.updateName.connect(self.saveUpdatedModel)
        
        # Start the thread
        
        if  len(train_data) == 0:
            self.message.setPlainText("CANNOT TRAIN DATA ARRAY OF LENGTH 0")
        else:
            self.trainThread.start()
            self.collect.setEnabled(False)
            self.train.setEnabled(False)
            self.test.setEnabled(False)
        
            # Reset tasks
            self.trainThread.finished.connect(
                lambda: self.collect.setEnabled(True)
                )
            self.trainThread.finished.connect(
                lambda: self.train.setEnabled(True)
                )
            self.trainThread.finished.connect(
                lambda: self.test.setEnabled(True)
                )


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = MainWin()
    sys.exit(app.exec_())
    