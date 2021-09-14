from hand_tracking import HandDetector
import cv2
import time

def main():
    # franerate
    previous_time = 0
    current_time = 0
    cam = cv2.VideoCapture(0)
    
    detector = HandDetector()
    
    while True:
        success, camera = cam.read()
        camera = detector.find_hands(camera)
        land_mark_list = detector.find_position(camera, draw = False)   # draw=False doesn't allow the thick circle on a finger to be shown
        if len(land_mark_list) != 0:
            print(land_mark_list[4])
        
        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time
        
        
        #                               position |                   3-> scale  | color    |  thickness-> 3
        cv2.putText(camera, str((fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow('Hand Detection', camera)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

