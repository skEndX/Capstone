import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
from sklearn.svm import SVC
from faceidentify.SVMclassifier import model as svm
from faceidentify.SVMclassifier import out_encoder

#### FOR GAZE AND MOTION ####
import time 
import argparse
import cv2
import os.path as osp
from detectheadposition import headpose
from gaze_tracking import GazeTracking

#### FOR WARNING ####
import pygame  # For play Sound
import time  # For sleep
import threading  # For multi thread
# from tkinter import *
# import tkinter.messagebox

# def Msgbox1():
#     tkinter.messagebox.showwarning("경고", "집중하세요")
import sys #data result send
# 결과 데이터 txt 파일 저장
f=open('C:/Capstone/result_data.txt','a')

# Warning Sound
import asyncio
import time


# UI import
import tkinter
from tkinter import ttk

from PIL import Image, ImageTk
import cv2

# 메세지 import
from tkinter import messagebox
 
####################### window 1
window = tkinter.Tk()
## window 창 설정 ########################################## 
# title
window.title("임베부스러기") # window 제목
# geomerty
window.geometry('640x500+100+100') # window 창 사이즈
# preventing GUI from resizing
window.resizable(False, False) # 창 조절 불가
#########################################################

def buttonClicked():
    label.configure()
    #labelNew.configure(text='잠시만 기다려주세요')
    #print(name.get())
    global name1, time1
    name1=name.get() #입력한 이름 받아오기
    time1=TestTime.get()
    

    print("이름과 시간")
    print(name1)
    print(time1)


    if(name1==""):

        nameLabel=labelNew.configure(font=("나눔고딕",15),text="이름을 확인해주세요")
        labelNew1.place(x=240,y=400)  

    elif(time1==""):
        nameLabel=labelNew.configure(font=("나눔고딕",15),text="시간을 확인해주세요")
        labelNew1.place(x=240,y=400)  

    else:
        nameLabel=labelNew.configure(font=("나눔고딕",15),text="잠시만 기다리세요")
        labelNew1.place(x=250,y=400)  

        

        messagebox.showinfo("확인","얼굴을 확인합니다") # 팝업 이름, 팝업 메세지

        main(args)



##################################
label=ttk.Label(window,font=("나눔고딕",23), text="시험 정보를 입력하세요")
label.place(x=160,y=150)

# Text Box
name , TestTime= tkinter.StringVar(),tkinter.StringVar()

# 이름을 넣는 칸
label=ttk.Label(window,font=("나눔고딕",15), text="이름")
label.place(x=200,y=228)
textbox1 = ttk.Entry(window, width=15, textvariable=name)
textbox1.place(x=265,y=230)

# 시간 넣는 칸
label=ttk.Label(window,font=("나눔고딕",15), text="시험 시간")
label.place(x=175,y=268)
textbox2 = ttk.Entry(window, width=15, textvariable=TestTime)
textbox2.place(x=265,y=270)



# Button
button1=button = ttk.Button(window,width=10, command=buttonClicked,text="확인")
button1.place(x=280,y=320)

# Label
labelNew1=labelNew = ttk.Label(window, text="")



def Sound():
    pygame.mixer.init()
    music = pygame.mixer.Sound("Warning/warning.wav")
    music.play()
    time.sleep(5)

# Settle Cheater
def Fail(timee, redcard):
    if redcard >= timee/3:
        print("===부정행위자 입니다===")
        f.write("   부정행위자(경고누적) \n")

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # print(face_pixels.shape)
    # transform face into one sample
    # expand dims adds a new dimension to the tensor
    samples = np.expand_dims(face_pixels, axis=0)
    # print(samples.shape)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

# Print Result


def PrintResult(x, y):
    print("###############--RESULT--#################")
    print("yellocard:", x, "/ redcard", y)
    print("###########################################")

# point can't get negative


def notnegative(x):
    if x < 0:
        return 0
    else:
        return x




def TxtOpen():
    messagebox.showinfo("결과출력","결과가 출력됩니다.")
    data = open('C:/Capstone/result_data.txt', 'r')
    contents = data.read()
    messagebox.showinfo("결과 출력",contents)
    f.close()




# main function
def main(args):
    filename = args["input_file"]
#<<<<<<< HEAD
    faceCascade = cv2.CascadeClassifier('C:/Capstone/models/haarcascade_frontalface_default.xml')
    model = load_model('C:\Capstone\poscas\POSCO_AIProject_OnlineTestCheatingDetectionAiSystem-master\models/facenet_keras.h5')
    #C:\Capstone\poscas\POSCO_AIProject_OnlineTestCheatingDetectionAiSystem-master\models
#=======
    faceCascade = cv2.CascadeClassifier(
        'models/haarcascade_frontalface_default.xml')
    model = load_model('models/facenet_keras.h5')
#>>>>>>> 757e3559f2acad057224670a45fca1fc2d17309e



    if filename is None:
        isVideo = False
        #url='http://192.168.0.06:8091/?action=stream'
        #webcam = cv2.VideoCapture(url)



        webcam = cv2.VideoCapture(0) # 캠으로 이미지 받아오는 코드
        webcam.set(3, args['wh'][0])
        webcam.set(4, args['wh'][1])       
    else:
        isVideo = True
        webcam = cv2.VideoCapture(filename)
        fps = webcam.get(cv2.webcam_PROP_FPS)
        width = int(webcam.get(cv2.webcam_PROP_FRAME_WIDTH))
        height = int(webcam.get(cv2.webcam_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)
        out = cv2.VideoWriter(args["output_file"],
                              fourcc, fps, (width, height))
        

    
    ##############################################################################################
   
    # 이름 제대로 들어가는지 테스트 -> 이름 제대로 들어감
    # print('main 문 ')
    # print(name1)
    # print(time1)

    # ##############################
    UserName = name1
    f.write(UserName+"   ")
    checktime = 1
    start_check= time.time() + (10 * checktime)
    checktime_end = time.time() + (60 * checktime) #1분 동안 체크 (60)
    while (webcam.isOpened()):# Infinity Loop for Detect Cheating for Online test

        ret, frame = webcam.read()  # Read wabcam
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)  # face structure
        for (x, y, w, h) in faces:
            # take the face pixels from the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # turn the face pixels back into an image
            crop_frame = frame[y:y+h, x:x+w]
            # resize the image to meet the size requirment of facenet
            new_crop = Image.fromarray(crop_frame)
            # turn the image back into a tensor
            new_crop = new_crop.resize((160, 160))
            # get the face embedding using the face net model
            crop_frame = np.asarray(new_crop)
            # it is a 1d array need to reshape it as a 2d tensor for svm
            face_embed = get_embedding(model, crop_frame)
            # predict using our SVM model
            face_embed = face_embed.reshape(-1, face_embed.shape[0])
            pred = svm.predict(face_embed)  # get the prediction probabiltiy
            # pred_prob has probabilities of each class
            pred_prob = svm.predict_proba(face_embed)

            # get name
            class_index = pred[0]
            class_probability = pred_prob[0, class_index] * 100
            predict_names = out_encoder.inverse_transform(pred)
            text = '%s (%.3f%%)' % (predict_names[0], class_probability)
            cv2.putText(frame, text, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('capstone', frame) #-> 정상 출력 되는 코드
        #label.after(20,frame) #-> window에 비디오 창이 열리게 끔 하고 싶었으나 안됨
    


        cv2.waitKey(1)   

        
        if(time.time() > start_check and predict_names[0]==UserName and class_probability > 80):
           
            #print("얼굴이 일치합니다. 시험을 시작하겠습니다.")
            messagebox.showinfo("얼굴 확인","얼굴이 일치합니다. 시험을 시작합니다")
            break 

       

        if time.time() > checktime_end:
            #print("얼굴이 일치하지 않아 시험에 응시하지 못합니다.")
            messagebox.showerror("얼굴 확인","얼굴이 일치하지 않아 시험에 응시하지 못합니다.")
            f.write("    얼굴 불일치"+'\n')
            

            TxtOpen() # 작동 함
            

            quit()
            window.destroy() # UI 화면 닫기
            break 
       
       ##################################################################################


       #########################################################################
    # Variable Setting
    hpd = headpose.HeadposeDetection(
        args["landmark_type"], args["landmark_predictor"])  # import headpose
    gaze = GazeTracking()  # import gazetracking
    yellocard = 0
    redcard = 0
    tempval = 0



    # Input time for limit test time
    timee=int(time1)
    #timee = int(input("시험 시간을 입력하세요(Minute): "))
    max_time_end = time.time() + (60 * timee)


    check_angle=time.time()+(10*checktime)
    while(webcam.isOpened()):# Infinity Loop for Detect Cheating for Online test
        ret, frame = webcam.read()  # Read wabcam
        gaze.refresh(frame)
        frame = gaze.annotated_frame()  # Mark pupil for frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)  # face structure

        # Get point from pupil
        if gaze.is_blinking():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        elif gaze.is_right():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        elif gaze.is_left():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        elif gaze.is_center():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        else:
            yellocard = yellocard + 2

        # Get redcard optiom
        if yellocard > 50:
            yellocard = 0
            tempval = tempval + 1
            redcard = redcard + 1

        # if get 1redcard, then give Aural and Text Warning(Loop)
        if tempval == 1:
            text1 = "WARNING"
            cv2.putText(frame, text1, (10, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255), 2)
            my_thread = threading.Thread(target=Sound)
            my_thread.start()
            tempval = 0

    # if you are not GPU environment, Do not run this code by # --------------
        # if get 2redcard, then give Picture Warning(Once)
        if redcard == 2:
            warn_img = cv2.imread("Warning/warning.png", cv2.IMREAD_COLOR)
            cv2.imshow('Warning', warn_img)
            cv2.waitKey(1)
            redcard = 2.1
    # -----------------------------s------------------------------------------
        # Get log consistently
        print("<< *의심수준:", yellocard, " || ", "*경고횟수:", redcard, " >>")
        #cv2.destroyWindow('Warning')
        # Detect head position
        if isVideo:
            frame, angles = hpd.process_image(frame)
            if frame is None: 
                break
            else:
                out.write(frame)
        else:
            frame, angles = hpd.process_image(frame)
            if angles is None:
                #print("경고! 응시자가 사라졌습니다")
                #messagebox.showwarning("경고","경고! 응시자가 사라졌습니다")
                if time.time() > check_angle:
                    redcard= timee/3+ redcard
                    #print("지속적으로 카메라 앵글 밖으로 나갔으므로, 시험을 강제종료합니다.")
                    #messagebox.showinfo("확인","얼굴이 일치합니다. 시험을 시작합니다")
                    messagebox.showerror("경고","지속적으로 카메라 앵글 밖으로 나갔으므로, 시험을 강제종료합니다.")
                   
                    PrintResult(yellocard, redcard)
                    Fail(timee, redcard)


                    TxtOpen()


                    window.destroy()
                    quit()
                    
                else:
                    pass
                
            else:  # angles = [x,y,z] , get point from headposition
                if angles[0] > 15 or angles[0] < -15 or angles[1] > 15 or angles[1] < -15 or angles[2] > 15 or angles[2] < -15:
                    yellocard = yellocard + 2
                else:
                    yellocard = yellocard - 1
                    yellocard = notnegative(yellocard)

        yellocard = yellocard + hpd.yello(frame)
        if yellocard < 0:
            yellocard = notnegative(yellocard)

        # Display the resulting frame
            cv2.imshow('capstone', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #print("관리자에 의해 시험이 강제 종료 되었습니다")

            f.write("경고 횟수 : %s" %(redcard))
            f.write("   -> 강제 종료 \n")

            messagebox.showerror("경고","관리자에 의해 시험이 강제 종료 되었습니다")
            
            PrintResult(yellocard, redcard)
            Fail(timee, redcard)



            # messagebox.showinfo("결과출력","결과가 출력됩니다.")
            # #messagebox("결과출력","결과가 출력됩니다.")
            # data = open('C:/Capstone/result_data.txt', 'r')
            # contents = data.read()
            # messagebox.showinfo("결과 출력",contents)


            TxtOpen()


            window.destroy()
          #  f.close()
            break
        elif time.time() > max_time_end:
            #print(timee, "분의 시험이 종료되었습니다.")
            f.write("    -> 정상 종료 \n")
            
            PrintResult(yellocard, redcard)
            Fail(timee, redcard)



            messagebox.showinfo("시험 종료","시험이 종료되었습니다.")
            
            #messagebox.showinfo("결과출력","결과가 출력됩니다.")
            # data = open('C:/Capstone/result_data.txt', 'r')
            # contents = data.read()
            # messagebox.showinfo("결과 출력",contents)
            
           # f.close()


            TxtOpen()


            window.destroy()
            break

    # When everything done, release the webcam
    webcam.release()
    cv2.destroyAllWindows()


    #TxtOpen() -> 새로운 UI 창이 뜸.. 여기에 넣으면 안될 듯


    quit()
    window.destroy()
    if isVideo:
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None,
                        help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file',
                        default=None, help='Output video.')
    parser.add_argument('-wh', metavar='N', dest='wh',
                        default=[720, 480], nargs=2, help='Frame size.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type',
                        type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor',
                        default='gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())




    window.mainloop()
 #<<<<<<< HEAD
    main(args)
 #=======
    main(args)


f.close()