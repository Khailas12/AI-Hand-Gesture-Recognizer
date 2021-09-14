import mediapipe as mp
import cv2
import time


class HandDetector():
    def __init__(
        self,
        static_image_mode = False,
        max_hands = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5):
        
        self.static_image_mode = static_image_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.static_image_mode, self.max_hands, self.min_detection_confidence, self.min_tracking_confidence)
        # # drawing a line and this code reduces the hard coding effort by the help of mediapipe
        self.mp_draw = mp.solutions.drawing_utils   
        
        
    def find_hands(self, camera, draw = True):
        img_rgb = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(camera)
        # print(results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(camera, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return camera
                

    def find_position(self, camera, hand_num=0, draw=True):
        
        land_mark_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                height, width, channels = camera.shape
                # positions, the x, y axis
                cx, cy = int(lm.x*width), int(lm.y*height)
                # print(id, cx, cy)
                land_mark_list.append([id, cx, cy])
            
            # if id == 4:     # 4 -> tip of the thumb
            if draw:
                cv2.circle(camera, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # for getting cx, cy info

        return land_mark_list
    

