import numpy as np
import math
import cv2
import os, sys
import traceback
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
import enchant
ddd=enchant.Dict("en-US")
hand_detector_1 = HandDetector(maxHands=1)
hand_detector_2 = HandDetector(maxHands=1)
import tkinter as tk
from PIL import Image, ImageTk
import gtts
import playsound
from tkinter import ttk

OFFSET = 29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"



class SignLanguageApp:

    def __init__(self):
        # Constructor for initializing the SignLanguageToTextConverter class.
        
        # - Initializes video capture from the default camera (index 0).
        # - Loads a pre-trained CNN model for sign language recognition.
        # - Sets up variables and flags for gesture recognition and tracking.
        # - Creates a Tkinter GUI window for displaying camera feed and results.
        # - Configures various GUI elements such as labels, buttons, and dropdown menus.
        # - Initializes variables for managing text conversion, speech synthesis, and UI updates.
       
        self.video_stream = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('/Users/macbook/Downloads/American-sign-Language-main/Final Project/Source Code/cnn8grps_rad1_model.h5')

        self.character_count = {}
        self.character_count['blank'] = 0
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = []
        for i in range(10):
            self.ten_prev_char.append(" ")

        for i in ascii_uppercase:
            self.character_count[i] = 0

        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.config(background='black')
        self.root.title("Multilingual SLR")
        self.root.protocol('WM_DELETE_WINDOW', self.close_app)
        self.root.geometry("1300x780")


        self.panel = tk.Label(self.root)
        self.panel.place(x=440, y=115, width=400, height=400)

        self.panel2 = tk.Label(self.root)  # initialize image panel
        self.panel2.place(x=25, y=115, width=400, height=400)

        self.T = tk.Label(self.root)
        self.T.place(x=589, y=5)
        self.T.config(text="Multilingual SLR",bg='black',fg='#149414', font=("Times New Roman", 24, "bold"))


        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=280, y=645)
        self.panel3.config(bg='black',fg='#149414')

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=645)
        self.T1.config(text="Character :", bg='black',fg='#149414',font=("Times New Roman", 30, "bold"))

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=280, y=688)
        self.panel5.config(bg='black',fg='#149414')


        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=688)
        self.T3.config(text="Sentence :",bg='black',fg='#149414', font=("Times New Roman", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=10, y=730)
        self.T4.config(text="Suggestions :", bg='black',fg='#149414', font=("Times New Roman", 30, "bold"))

        self.button1 = tk.Button(self.root)
        self.button1.place(x=390, y=730)

        self.button2 = tk.Button(self.root)
        self.button2.place(x=590, y=730)

        self.button3 = tk.Button(self.root)
        self.button3.place(x=790, y=730)

        self.button4 = tk.Button(self.root)
        self.button4.place(x=990, y=730)

        self.clear = tk.Button(self.root)
        self.clear.place(x=1200, y=630)
        self.clear.config(text="Clear", font=("Times New Roman", 20,'bold'),bg='#149414', wraplength=100, command=self.clear_interface)

        self.speak = tk.Button(self.root)
        self.speak.place(x=1085, y=630)
        self.speak.config(text="Speak", font=("Times New Roman", 20,'bold'),bg='#149414', wraplength=100, command=self.speak_text)

        self.current_sentence = " "
        self.counter = 0
        self.current_word = " "
        self.current_character = "C"
        self.photo = "Empty"

        self.suggested_word_1 = " "
        self.suggested_word_2 = " "
        self.suggested_word_3 = " "
        self.suggested_word_4 = " "
        options = ["Australian", "British", "Indian"]
        self.dropdown = ttk.Combobox(self.root, values=options, state="readonly")
        self.dropdown.current(0)
        self.dropdown.place(x=1000, y=50)

        self.video_loop()

    
    def video_loop(self):
        
        # Continuously captures frames from the camera, processes them, and updates the GUI.

        # - Reads a frame from the camera and flips it horizontally.
        # - Detects and tracks hands in the flipped frame using HandDetector.
        # - Draws landmarks and lines on a white canvas based on hand keypoints.
        # - Resizes and displays the processed image on the GUI panel.
        # - Predicts the sign language gesture using the pre-trained CNN model.
        # - Updates GUI elements such as labels, buttons, and images with gesture information.
        # - Handles exceptions and continues the loop.
        # - Calls itself recursively to maintain real-time video processing.
    
        try:
            ok, frame = self.video_stream.read()
            cv2image = cv2.flip(frame, 1)
            hands = hand_detector_1.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy=np.array(cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                image = cv2image_copy[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]

                white = cv2.imread("./white.jpg")

                handz = hand_detector_2.findHands(image, draw=False, flipType=True)
                print(" ", self.counter)
                self.counter += 1
                if handz:
                    hand = handz[0]
                    self.pts = hand['lmList']
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    for t in range(0, 4, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                             (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                             3)

                    for i in range(21):
                        cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                    res=white
                    self.predict(res)

                    self.current_image2 = Image.fromarray(res)

                    imgtk = ImageTk.PhotoImage(image=self.current_image2)

                    self.panel2.imgtk = imgtk
                    self.panel2.config(image=imgtk)

                    image_path = f"/Users/macbook/Downloads/American-sign-Language-main/Final Project/Source Code/Images/{self.dropdown.get()}/{self.current_charactergit}.jpg"
                    image1 = Image.open(image_path)
                    image1 = image1.resize((200, 200), Image.LANCZOS)
                    test = ImageTk.PhotoImage(image1)
                    label1 = tk.Label(image=test)
                    label1.image = test
                    label1.place(x=1000, y=110)

                    self.panel3.config(text=self.current_character, font=("Times New Roman", 30))

                    self.button1.config(text=self.suggested_word_1, font=("Times New Roman", 20), wraplength=825, command=self.action1)
                    self.button2.config(text=self.suggested_word_2, font=("Times New Roman", 20), wraplength=825,  command=self.action2)
                    self.button3.config(text=self.suggested_word_3, font=("Times New Roman", 20), wraplength=825,  command=self.action3)
                    self.button4.config(text=self.suggested_word_4, font=("Times New Roman", 20), wraplength=825,  command=self.action4)

            self.panel5.config(text=self.current_sentence, font=("Times New Roman", 30), wraplength=1025)
        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def calculate_distance(self,x,y):
        # Calculates the Euclidean distance between two points in 2D space.
        # 
        # - Receives two points (x and y) as input.
        # - Computes the Euclidean distance between the points using the distance formula.
        # - Returns the distance between the points as a floating-point value.
        # 
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def clear_interface(self):
        # Resets various attributes and UI elements to their initial states for clearing the interface.
        # 
        # - Resets the string representation of the sentence and related variables.
        # - Resets the counter for hand tracking.
        # - Resets the current word and symbol.
        # - Resets the photo attribute.
        # - Resets word suggestions to empty strings.
        # - Clears the display panel for the current symbol and sentence.

        self.current_sentence = " "
        self.counter = 0
        self.current_word = " "
        self.current_character = "C"
        self.photo = "Empty"
        self.suggested_word_1 = " "
        self.suggested_word_2 = " "
        self.suggested_word_3 = " "
        self.suggested_word_4 = " "
        self.panel3.config(text=" ", font=("Courier", 50))
        self.panel5.config(text=" ", font=("Courier", 50))

    def close_app(self):
        # Destroys the application, releasing resources and closing windows.
        # 
        # - Prints a closing message to the console.
        # - Destroys the Tkinter root window.
        # - Releases the video stream.
        # - Closes all OpenCV windows.
        
        print("Closing Application...")
        self.root.destroy()
        self.video_stream.release()
        cv2.destroyAllWindows()



    def predict(self, test_image):
        # The "test image" refers to the frame captured by the camera, containing the vertices of the hand. It provides us with an image of the hand, as depicted in the central section of the gui.
        white=test_image
        # Reshape the input test image
        white = white.reshape(1, 400, 400, 3)


        # Perform prediction using the model returns a numpy array contains the values
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        
        # Extract the most probable classes returns the index of the largest value from the numpy array
        ch1 = np.argmax(prob, axis=0)
        # override the largest value 0
        prob[ch1] = 0
        # find the index of the largest value again and it will be the second largest value
        ch2 = np.argmax(prob, axis=0)

        pl = [ch1, ch2]

        # Define conditions for each class
        # Here, we have a list 'l' containing pairs of indices representing certain classes.
        # We check if the pair 'pl' (ch1, ch2) matches any of these predefined pairs in 'l'.
        # If it does, we further check specific conditions to determine the class of the character.
        # The conditions involve comparing the y-coordinates of certain points in 'self.pts'.
        # If all conditions are met, we set ch1 to 0, indicating the predicted class.
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        
        # Check if the predicted class pair 'pl' is in the predefined list 'l'
        if pl in l:
              # Further check specific conditions based on the position of hand landmarks
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                # If all conditions are met, set ch1 to 0
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2
                # print("22222")

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.calculate_distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2
                # print("22222")


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3
                print("33333c")



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3
                print("33333b")

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3
                print("33333a")

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.calculate_distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4
                # print("44444")

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.calculate_distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4
                # print("44444")

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
                # print("44444")

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5
                print("55555b")

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5
                print("55555a")

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5
                # print("55555")

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5
                # print("55555")

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7
                # print("77777")

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7
                # print("77777")

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7
                # print("77777")

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6
                print("666661")


        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6
                print("666662")

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.calculate_distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6
                print("666663")

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.calculate_distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6
                print("666664")

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6
                print("666665")

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1
                print("111111")

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
                print("111112")

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
                print("111112")

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1
                print("111113")

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.calculate_distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1
                print("1111993")

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1
                print("1111mmm3")

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1
                print("1111140")

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1
                print("111114")

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7
                print("111114lll;;p")

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1
                print("111115")

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.calculate_distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1
                print("111116")

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1
                print("1117")

        # -------------------------conditions for 8 groups  ends

        # -------------------------conditions for subgroups  starts
        #
        # Assign character label based on ch1 value
        if ch1 == 0:
            # If ch1 is 0, initially assigned as 'S'
            ch1 = 'S'
            # Check specific conditions to determine the character based on hand landmarks
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                 # If conditions met, update ch1 to 'A'
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                 # If conditions met, update ch1 to 'T'
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                 # If conditions met, update ch1 to 'E'
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                 # If conditions met, update ch1 to 'M'
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                 # If conditions met, update ch1 to 'N'
                ch1 = 'N'

        if ch1 == 2:
            if self.calculate_distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.calculate_distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.calculate_distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.calculate_distance(self.pts[8], self.pts[12]) - self.calculate_distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.calculate_distance(self.pts[8], self.pts[12]) - self.calculate_distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1=" "



        print(self.pts[4][0] < self.pts[5][0])
        if ch1 == 'E' or ch1=='Y' or ch1=='B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1="next"


        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'


        if ch1=="next" and self.prev_char!="next":
            if self.ten_prev_char[(self.count-2)%10]!="next":
                if self.ten_prev_char[(self.count-2)%10]=="Backspace":
                    self.current_sentence=self.current_sentence[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.current_sentence = self.current_sentence + self.ten_prev_char[(self.count-2)%10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.current_sentence = self.current_sentence + self.ten_prev_char[(self.count - 0) % 10]

        # Check if ch1 is a space character and the previous character is not a space
        if ch1 == " " and self.prev_char != " ":
            # If conditions met, add a space to the string
            self.current_sentence = self.current_sentence + " "

        # Update the previous character with the current character
        self.prev_char = ch1
        # Update the current symbol with the current character
        self.current_character = ch1
        # Increment the count to keep track of characters processed
        self.count += 1
        # Update the circular buffer storing the ten previous characters
        self.ten_prev_char[self.count % 10] = ch1


        # Check if the string contains any non-space characters
        if len(self.current_sentence.strip()) != 0:
            # Find the index of the last space character
            st = self.current_sentence.rfind(" ")
            # Get the substring after the last space character
            ed = len(self.current_sentence)
            word = self.current_sentence[st + 1:ed]
            # Store the extracted word
            self.current_word = word
            print("----------word = ", word)
            # If the extracted word contains non-space characters
            if len(word.strip()) != 0:
                # Spell check the word and retrieve suggestions
                ddd.check(word)
                lenn = len(ddd.suggest(word))
                # Store up to four suggestions if available
                if lenn >= 4:
                    self.suggested_word_4 = ddd.suggest(word)[3]

                if lenn >= 3:
                    self.suggested_word_3 = ddd.suggest(word)[2]

                if lenn >= 2:
                    self.suggested_word_2 = ddd.suggest(word)[1]

                if lenn >= 1:
                    self.suggested_word_1 = ddd.suggest(word)[0]
            # If the extracted word is empty, set suggestions to empty strings
            else:
                self.suggested_word_1 = " "
                self.suggested_word_2 = " "
                self.suggested_word_3 = " "
                self.suggested_word_4 = " "

        # Initialize the HandDetector object with specified parameters
        hd = HandDetector(detectionCon=0.7, maxHands=1)
        # Print a loading message indicating the application is being loaded
        print("Application Loading...")

    def action1(self):
        # Find the index of the last space in the string
        idx_space = self.current_sentence.rfind(" ")
        # Find the index of the current word in the string starting from the last space
        idx_word = self.current_sentence.find(self.current_word, idx_space)
        # Get the index of the last character in the string
        last_idx = len(self.current_sentence)
        # Remove the current word from the string and replace it with the first suggested word in uppercase
        self.current_sentence = self.current_sentence[:idx_word]
        self.current_sentence = self.current_sentence + self.suggested_word_1.upper()

    def action2(self):
        # Find the index of the last space in the string
        idx_space = self.current_sentence.rfind(" ")
        # Find the index of the current word in the string starting from the last space
        idx_word = self.current_sentence.find(self.current_word, idx_space)
        # Get the index of the last character in the string
        last_idx = len(self.current_sentence)
        # Remove the current word from the string and replace it with the second suggested word in uppercase
        self.current_sentence=self.current_sentence[:idx_word]
        self.current_sentence=self.current_sentence+self.suggested_word_2.upper()

    def action3(self):
        # Find the index of the last space in the string
        idx_space = self.current_sentence.rfind(" ")
        # Find the index of the current word in the string starting from the last space
        idx_word = self.current_sentence.find(self.current_word, idx_space)
        # Get the index of the last character in the string
        last_idx = len(self.current_sentence)
        # Remove the current word from the string and replace it with the third suggested word in uppercase
        self.current_sentence = self.current_sentence[:idx_word]
        self.current_sentence = self.current_sentence + self.suggested_word_3.upper()

    def action4(self):
        # Find the index of the last space in the string
        idx_space = self.current_sentence.rfind(" ")
        # Find the index of the current word in the string starting from the last space
        idx_word = self.current_sentence.find(self.current_word, idx_space)
        # Get the index of the last character in the string
        last_idx = len(self.current_sentence)
        # Remove the current word from the string and replace it with the fourth suggested word in uppercase
        self.current_sentence = self.current_sentence[:idx_word]
        self.current_sentence = self.current_sentence + self.suggested_word_4.upper()


    def speak_text(self):
        text= self.current_sentence
        sound=gtts.gTTS(text,lang='en')
        sound.save('prediction.mp3')
        playsound.playsound('prediction.mp3')
    






app = SignLanguageApp()
# Start the main event loop of the tkinter application to display the GUI
app.root.mainloop()
