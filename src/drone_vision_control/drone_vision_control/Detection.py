from ultralytics import YOLO
from collections import defaultdict
import cv2
from drone_vision_control import Recognition
import rclpy
from rclpy.node import Node
import supervision as sv
import math
import numpy as np


class Detection:
    def __init__(self, matcher):
        """Initialization of a drone detection instance
        
        Args: matcher - matcher
        """
        self.matcher=matcher
        #Track history
        self.track_history=defaultdict(lambda:[])
        self.MAX_TRACK_LENGTH=300
        self.FRAME_TO_REFRESH=100
        

    def setup_detector(self, model_path, classes_to_detect):
        """
        Method to setup the detection model and detection classes
        
        Args: model_path - path to the YOLO model
            classes_to_detect - list of a classes to detect

        Returns:
            model - specified model and selected classes
        """
        self.model_path=model_path
        self.classes_to_detect=classes_to_detect
        self.model=YOLO(self.model_path)
        self.classes=[]

        for id, name in self.model.names.items():
            if name in classes_to_detect:
                self.classes.append(id)

        return self.model

    def track(self,frame):
        """
        Method to track using YOLO
        
        Args: frame - frame from the camera

        Returns: results - tracking results
        """
        self.frame=frame
        self.results=self.model.track(self.frame,persist=True, tracker="bytetrack.yaml", classes=self.classes)

        #SORT FIX
        # self.detections=self.results[0].boxes

        #  # Convert detections to [x1, y1, x2, y2, confidence] for SORT
        # sort_input = []
        # for det in self.detections:
        #     x1, y1, x2, y2 = det.xyxy[0]
        #     conf = det.conf[0]
        #     sort_input.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])

        # # Update SORT tracker
        # tracks = self.sort_tracker.update(np.array(sort_input))

        #  # Optionally annotate
        # if not self.detections[0].boxes:
        #     cv2.putText(self.frame, "SORT SORT :",(20,630), 2, 1.0, (0,255,0), 2)

        #     return tracks
        
        #SORT FIX

        return self.results[0]
    

    def center_detected_body(self, box, track_id, depth_image, pose_model,prev_errors, integrals,dt,name,state_value):
        """
        Method to center the detected person body in the frame if it is recognized.

        Args:box - bounding box of detected object
            track_id - track IDs of detected object
            depth_image - depth image from realsesne camera
            pose_model - YOLO-pose model
            prev_errors - previous errors stored for PID regulator
            integrals - integrals stored for PID regulator
            dt - time interval
            name - stored name of a tracked person
            state_value - state value for drone color manegement

        Returns:frame - annotated frame
            cmd_linear_x - drone cmd_vel linear command in x direction
            cmd_angular_z - drone cmd_vel angular command in z direction
            cmd_linear_z - drone cmd_vel linear command in z direction
            prev_errors - stored errors in this loop for the next iteration
            integrals - stored integrals in this loop for the next iteration
            state_value - state value for drone color management
        """

        self.box=box
        self.track_id=track_id
        self.depth_image=depth_image
        self.pose_model=pose_model  
        self.prev_errors=prev_errors
        self.dt=dt
        self.integrals=integrals
        self.name=name
        self.state_value=state_value
        #YOLOv8 person bounding box
        x,y,w,h=self.box     #x,y (box center), width, height
        yolo_box=int(x-w/2),int(y-h/2),int(w),int(h)    #x,y (top left), width, height

        track=self.track_history[self.track_id]
        track.append((float(x), float(y)))
        if len(track)>self.MAX_TRACK_LENGTH:
                    track.pop(0)
                
        #Detect pose in the sepcified bounding box
        self.pose_results=self.pose_model(self.frame)

        #Check if the results are obtrained
        if self.pose_results and len(self.pose_results) > 0:
            #Extract the keypoints from the first result
            self.pose_keypoints=sv.KeyPoints.from_ultralytics(self.pose_results[0])
            for i in range(len(self.pose_keypoints)):
                #Check if any keypoints are obtained
                if len(self.pose_keypoints) > 0:
                    self.keypoint_names=StoreKeypoints(keypoint_detections=self.pose_keypoints,i=i)

                #Top of the chest keypoint (center between shoulders )
                self.top_center_body_keypoint=((self.keypoint_names.LEFT_SHOULDER[0]+self.keypoint_names.RIGHT_SHOULDER[0])/2,
                                            (self.keypoint_names.LEFT_SHOULDER[1]+self.keypoint_names.RIGHT_SHOULDER[1])/2)
                #Center between hips
                self.bottom_center_body_keypoint=((self.keypoint_names.LEFT_HIP[0]+self.keypoint_names.RIGHT_HIP[0])/2,
                                                (self.keypoint_names.LEFT_HIP[1]+self.keypoint_names.RIGHT_HIP[1])/2)
                
                #Calculate chest depth distance followind the line from chest to hips
                # Number of points to sample along the line
                num_samples = 20
                # Calculate the depth values along the line and store them
                depth_values = []
                for i in range(num_samples):
                    # Linearly interpolate between the top-center and bottom-center keypoints
                    x_body = int(self.top_center_body_keypoint[0] + i * (self.bottom_center_body_keypoint[0] - self.top_center_body_keypoint[0]) / (num_samples - 1))
                    y_body = int(self.top_center_body_keypoint[1] + i * (self.bottom_center_body_keypoint[1] - self.top_center_body_keypoint[1]) / (num_samples - 1))

                    # Check if the interpolated point is within the depth image boundaries
                    if 0 <= x_body < self.depth_image.shape[1] and 0 <= y_body < self.depth_image.shape[0]:
                        # Append the depth at this point (convert to cm by dividing by 10)
                        depth_values.append(self.depth_image[y_body, x_body] / 10.0)

                # Calculate the average depth distance along the line, if we have valid depth values
                if depth_values:
                    self.depth_distance = sum(depth_values) / len(depth_values)

                #Check if the top_center_keypoint is inside the yolo bounding box   
                if self.top_center_body_keypoint[0] > yolo_box[0] and self.top_center_body_keypoint[0] < yolo_box[0]+yolo_box[2] and self.top_center_body_keypoint[1] > yolo_box[1] and self.top_center_body_keypoint[1] < yolo_box[1]+yolo_box[3]:
                
                    #The situation 1:
                    # where both top and bottom body keypoints are visible
                    if self.top_center_body_keypoint != (0.0,0.0) and self.bottom_center_body_keypoint != (0.0,0.0) and self.keypoint_names.LEFT_SHOULDER[0] != 0.0 and self.keypoint_names.LEFT_SHOULDER[1] != 0.0 and self.keypoint_names.RIGHT_SHOULDER[0] != 0.0 and self.keypoint_names.RIGHT_SHOULDER[1] != 0.0 and self.keypoint_names.LEFT_HIP[0] != 0.0 and self.keypoint_names.LEFT_HIP[1] != 0.0 and self.keypoint_names.RIGHT_HIP[0] != 0.0 and self.keypoint_names.RIGHT_HIP[1] != 0.0:
                            
                        #Calculate distance between two points(in pixels): sqrt((x2 -x1)**2 + (y2 -y1)**2)
                        self.body_keypoints_distance=math.sqrt(((self.bottom_center_body_keypoint[0]-self.top_center_body_keypoint[0])**2)+((self.bottom_center_body_keypoint[1]-self.top_center_body_keypoint[1])**2))
                        
                        #The distance calculated using YOLOv8-pose
                        self.pose_calculated_distance=int(0.00978*self.body_keypoints_distance**2-5.472*self.body_keypoints_distance+949.9) 
                        
                        if self.depth_distance < self.pose_calculated_distance*0.25 or self.depth_distance > self.pose_calculated_distance*4:
                            self.depth_distance=self.pose_calculated_distance

                        #Fusion of given distances (in cm)
                        self.fused_distance= (self.depth_distance + self.pose_calculated_distance)/2
                   
                        cv2.rectangle(self.frame, (int(x)-int(float(w)/2), int(y)-int(float(h)/2)-30),(int(x)-int(float(w)/2)+140, int(y)-int(float(h)/2)), (255,207,137),-1)
                        cv2.putText(self.frame,self.name, (int(x)-int(float(w)/2)+2, int(y)-int(float(h)/2)-4),2,1.0,(255,255,255),2)
                        cv2.rectangle(self.frame, (int(x)-int(float(w)/2), int(y)-int(float(h)/2)), (int(x)+int(float(w)/2), int(y)+int(float(h)/2)), (255,207,137), 2)
                        cv2.circle(self.frame, (int(self.top_center_body_keypoint[0]), int(self.top_center_body_keypoint[1])), 6,(255,255,255),-1)
                        cv2.putText(self.frame, "Distance: " + str(int(self.fused_distance)) + "cm",(20,700), 2, 1.0, (255,255,255), 2)
                        cv2.putText(self.frame, "FACE DETECTED:",(20,440), 2, 1.0, (200,200,200), 2)
                        cv2.putText(self.frame, self.name,(0,480), 2, 1.0, (255,255,255), 2)

                        #PID regulator(x movement):
                        #Left to right
                        if not(self.top_center_body_keypoint[0] > self.frame.shape[1]/2-8 and self.top_center_body_keypoint[0] < self.frame.shape[1]/2+8):
                            #PID gains
                            Kp1=0.0012 #0.0018
                            Ki1=0.0001
                            Kd1=0.0004

                        
                            error1=(self.frame.shape[1]/2-self.top_center_body_keypoint[0])
                            #P
                            p1=Kp1*error1
                            #I
                            self.integrals[0]+=Ki1*error1*self.dt
                            #D
                            d1=Kd1*(error1-self.prev_errors[0])/self.dt

                            self.cmd_msg_angular_z=p1+self.integrals[0]
                            self.cmd_msg_angular_z=p1+d1 

                            #Restrict the angular z:
                            if self.cmd_msg_angular_z > 0.8:
                                self.cmd_msg_angular_z = 0.8
                            if self.cmd_msg_angular_z < -0.8:
                                self.cmd_msg_angular_z = -0.8

                            #Store the error
                            self.prev_errors[0]=error1

                        else:
                            self.cmd_msg_angular_z=0.0
                        

                        #Store the error
                        self.prev_errors[1]=0.0
                        self.cmd_msg_linear_z=0.0

                
                        #Store the error
                        self.prev_errors[2]=0.0
                        
                    
                        self.cmd_msg_linear_x=0.0
                    else:
                        cv2.putText(self.frame, "Body NOT detected :",(20,630), 2, 1.0, (0,255,0), 2)
                        self.cmd_msg_linear_x=0.0
                        self.cmd_msg_linear_z=0.0
                        self.cmd_msg_angular_z=0.0
                        self.state_value=1700
                else:
                    cv2.putText(self.frame, "Body NOT detected :",(20,630), 2, 1.0, (0,255,0), 2)
                    self.state_value=1700   
        else:
            cv2.putText(self.frame, "Body NOT detected :",(20,630), 2, 1.0, (0,255,0), 2)
            self.cmd_msg_linear_x=0.0
            self.cmd_msg_linear_z=0.0
            self.cmd_msg_angular_z=0.0
            self.state_value=1700

        return self.frame,self.cmd_msg_linear_x, self.cmd_msg_angular_z, self.cmd_msg_linear_z, self.prev_errors, self.integrals,self.state_value


    def track_detected_body(self, box, track_id, depth_image, pose_model,prev_errors, integrals, dt,name, radius_state,state_value):
        """
        Method to track the detected person body in the frame if it is recognized.

        Args:box - bounding box of detected object
            track_id - track IDs of detected object
            depth_image - depth image from realsesne camera
            pose_model - YOLO-pose model
            prev_errors - previous errors stored for PID regulator
            integrals - integrals stored for PID regulator
            dt - time interval
            name - stored name of tracked individual
            state_value - state value for drone color management

        Returns:frame - annotated frame
            cmd_linear_x - drone cmd_vel linear command in x direction
            cmd_angular_z - drone cmd_vel angular command in z direction
            cmd_linear_z - drone cmd_vel linear command in z direction
            prev_errors - stored errors in this loop for the next iteration
            integrals - stored integrals in this loop for the next iteration
            state_value - state value for drone color management
        """

        self.box=box
        self.track_id=track_id
        self.depth_image=depth_image
        self.pose_model=pose_model  
        self.prev_errors=prev_errors
        self.integrals=integrals
        self.dt=dt
        self.name=name
        self.radius_state=radius_state
        self.state_value=state_value

        #YOLOv8 person bounding box
        x,y,w,h=self.box     #x,y (box center), width, height
        yolo_box=int(x-w/2),int(y-h/2),int(w),int(h)    #x,y (top left), width, height

        track=self.track_history[self.track_id]
        track.append((float(x), float(y)))
        if len(track)>self.MAX_TRACK_LENGTH:
                    track.pop(0)
        
        #Detect pose in the sepcified bounding box
        self.pose_results=self.pose_model(self.frame)

        #Check if the results are obtrained
        if self.pose_results and len(self.pose_results) > 0:
            #Extract the keypoints from the first result
            self.pose_keypoints=sv.KeyPoints.from_ultralytics(self.pose_results[0])
            for i in range(len(self.pose_keypoints)):
                #Check if any keypoints are obtained
                if len(self.pose_keypoints) > 0:
                    self.keypoint_names=StoreKeypoints(keypoint_detections=self.pose_keypoints,i=i)

                #Top of the chest keypoint (center between )
                self.top_center_body_keypoint=((self.keypoint_names.LEFT_SHOULDER[0]+self.keypoint_names.RIGHT_SHOULDER[0])/2,
                                            (self.keypoint_names.LEFT_SHOULDER[1]+self.keypoint_names.RIGHT_SHOULDER[1])/2)
                #Center between hips
                self.bottom_center_body_keypoint=((self.keypoint_names.LEFT_HIP[0]+self.keypoint_names.RIGHT_HIP[0])/2,
                                                (self.keypoint_names.LEFT_HIP[1]+self.keypoint_names.RIGHT_HIP[1])/2)
                #Calculate chest depth distance
                # Number of points to sample along the line
                num_samples = 20
                # Calculate the depth values along the line and store them
                depth_values = []
                for i in range(num_samples):
                    # Linearly interpolate between the top-center and bottom-center keypoints
                    x_body = int(self.top_center_body_keypoint[0] + i * (self.bottom_center_body_keypoint[0] - self.top_center_body_keypoint[0]) / (num_samples - 1))
                    y_body = int(self.top_center_body_keypoint[1] + i * (self.bottom_center_body_keypoint[1] - self.top_center_body_keypoint[1]) / (num_samples - 1))

                    # Check if the interpolated point is within the depth image boundaries
                    if 0 <= x_body < self.depth_image.shape[1] and 0 <= y_body < self.depth_image.shape[0]:
                        # Append the depth at this point (convert to cm by dividing by 10)
                        depth_values.append(self.depth_image[y_body, x_body] / 10.0)

                # Calculate the average depth distance along the line, if we have valid depth values
                if depth_values:
                    self.depth_distance = sum(depth_values) / len(depth_values)
                    #cv2.putText(self.frame, "Distance D: " + str(self.depth_distance) + "cm",(20,140), 2, 1.0, (255,255,255), 2)

            
                #Check if the top_center_keypoint is inside the yolo bounding box
                if self.top_center_body_keypoint[0] > yolo_box[0] and self.top_center_body_keypoint[0] < yolo_box[0]+yolo_box[2] and self.top_center_body_keypoint[1] > yolo_box[1] and self.top_center_body_keypoint[1] < yolo_box[1]+yolo_box[3]:
                    
                    # The situation 1:
                    # Both top and bottom body keypoints are visible
                    if self.top_center_body_keypoint != (0.0,0.0) and self.bottom_center_body_keypoint != (0.0,0.0) and self.keypoint_names.LEFT_SHOULDER[0] != 0.0 and self.keypoint_names.LEFT_SHOULDER[1] != 0.0 and self.keypoint_names.RIGHT_SHOULDER[0] != 0.0 and self.keypoint_names.RIGHT_SHOULDER[1] != 0.0 and self.keypoint_names.LEFT_HIP[0] != 0.0 and self.keypoint_names.LEFT_HIP[1] != 0.0 and self.keypoint_names.RIGHT_HIP[0] != 0.0 and self.keypoint_names.RIGHT_HIP[1] != 0.0:
                        
                        #Calculate distance between two points (in pixels): sqrt((x2 -x1)**2 + (y2 -y1)**2)
                        self.body_keypoints_distance=math.sqrt(((self.bottom_center_body_keypoint[0]-self.top_center_body_keypoint[0])**2)+((self.bottom_center_body_keypoint[1]-self.top_center_body_keypoint[1])**2))
                        #The distance calculated using YOLOv8-pose
                        self.pose_calculated_distance= int(0.00978*self.body_keypoints_distance**2-5.472*self.body_keypoints_distance+949.9)

                        if self.depth_distance < self.pose_calculated_distance*0.25 or self.depth_distance > self.pose_calculated_distance*4:
                            self.depth_distance=self.pose_calculated_distance

                        #Fusion of given distances (in cm)
                        self.fused_distance= (self.depth_distance + self.pose_calculated_distance)/2
                        

                        cv2.rectangle(self.frame, (int(x)-int(float(w)/2), int(y)-int(float(h)/2)-30),(int(x)-int(float(w)/2)+140, int(y)-int(float(h)/2)), (255,207,137),-1)
                        cv2.putText(self.frame,self.name, (int(x)-int(float(w)/2)+2, int(y)-int(float(h)/2)-4),2,1.0,(255,255,255),2)
                        cv2.rectangle(self.frame, (int(x)-int(float(w)/2), int(y)-int(float(h)/2)), (int(x)+int(float(w)/2), int(y)+int(float(h)/2)), (255,207,137), 2)
                        cv2.circle(self.frame, (int(self.top_center_body_keypoint[0]), int(self.top_center_body_keypoint[1])), 6,(255,255,255),-1)
                        cv2.putText(self.frame, "Distance : " + str(int(self.fused_distance)) + "cm",(20,700), 2, 1.0, (255,255,255), 2)
                        cv2.putText(self.frame, "FACE DETECTED:",(20,440), 2, 1.0, (200,200,200), 2)
                        cv2.putText(self.frame, self.name,(20,480), 2, 1.0, (255,255,255), 2)


                        #PID regulator(x movement):
                        #Left to right
                        if not(self.top_center_body_keypoint[0] > self.frame.shape[1]/2-8  and self.top_center_body_keypoint[0] < self.frame.shape[1]/2+8 ):
                            #PID gains
                            Kp1=0.0012 #0.0018
                            Ki1=0.0001
                            Kd1=0.0004

                        
                            error1=(self.frame.shape[1]/2-self.top_center_body_keypoint[0])
                            #P
                            p1=Kp1*error1
                            #I
                            self.integrals[0]+=Ki1*error1*self.dt
                            #D
                            d1=Kd1*(error1-self.prev_errors[0])/self.dt

                            self.cmd_msg_angular_z=p1+self.integrals[0]+d1 

                            #Restrict the angular z:
                            if self.cmd_msg_angular_z > 0.8:
                                self.cmd_msg_angular_z = 0.8
                            if self.cmd_msg_angular_z < -0.8:
                                self.cmd_msg_angular_z = -0.8

                            #Store the error
                            self.prev_errors[0]=error1

                        else:
                            self.cmd_msg_angular_z=0.0
                            self.integrals[0]=0.0
                        #Top to bottom
                        if not(self.top_center_body_keypoint[1] >= self.frame.shape[0]/2-10 and self.top_center_body_keypoint[1] <=self.frame.shape[0]/2-10):
                            #PID gains
                            Kp2=0.84 #0.8498
                            Ki2=0.0
                            Kd2=(0.336*9.446)/(1+9.446)

                            error2=(self.frame.shape[0]/2-self.top_center_body_keypoint[1])/600
                            #P
                            p2=Kp2*error2
                            #I
                            self.integrals[1]+=Ki2*error2*self.dt
                            #D
                            d2=Kd2*(error2-self.prev_errors[1])/self.dt

                            self.cmd_msg_linear_z=Kp2*error2+Ki2*self.integrals[1]

                            #Store the error
                            self.prev_errors[1]=error2
                            
                            #Override
                            self.cmd_msg_linear_z=0.0
                            
                        else:
                            self.cmd_msg_linear_z=0.0
                            self.integrals[1]=0.0

                        #X axis
                        if not(self.fused_distance >= (self.radius_state -10) and self.fused_distance <= (self.radius_state +10)):
                            #PID gains
                            Kp3=0.8004 #1.1004
                            Ki3=0.0
                            Kd3=(0.0746*3393.8)/(1+3393.8)
                            #print(f"Radius state: {self.radius_state}")

                            error3=(self.fused_distance-self.radius_state)/120
                            #P
                            p3=Kp3*error3
                            #I
                            self.integrals[2]+=Ki3*error3*self.dt
                            #D
                            d3=Kd3*(error3-self.prev_errors[2])/self.dt

                            self.cmd_msg_linear_x=Kp3*error3+Ki3*self.integrals[2]+Kd3*d3

                        

                            if self.cmd_msg_linear_x > 0.6:
                                self.cmd_msg_linear_x = 0.6
                            if self.cmd_msg_linear_x < -0.6:
                                self.cmd_msg_linear_x = -0.6
                            #Store the error
                            self.prev_errors[2]=error3
                            
                        else:
                            self.cmd_msg_linear_x=0.0
                            self.integrals[2]=0.0


                        #Watch out: if the point is not visible in the frame, the system by default sets the point values to 0
                    elif self.keypoint_names.LEFT_SHOULDER[0] != 0.0 and self.keypoint_names.LEFT_SHOULDER[1] != 0.0 and self.keypoint_names.RIGHT_SHOULDER[0] != 0.0 and self.keypoint_names.RIGHT_SHOULDER[1] != 0.0:
                        cv2.circle(self.frame, (int(self.top_center_body_keypoint[0]), int(self.top_center_body_keypoint[1])), 8,(255,255,255),-1)
                        #PID regulator(x movement):
                        #The center: (480,360)
                        #Left to right
                        if not(self.top_center_body_keypoint[0] > self.frame.shape[1]/2-8 and self.top_center_body_keypoint[0] < self.frame.shape[1]/2+8):
                            #PID gains
                            Kp1=0.0012
                            Ki1=0.0001
                            Kd1=0.0004

                        
                            error1=(self.frame.shape[1]/2-self.top_center_body_keypoint[0])
                            #P
                            p1=Kp1*error1
                            #I
                            self.integrals[0]+=Ki1*error1*self.dt
                            #D
                            d1=Kd1*(error1-self.prev_errors[0])/self.dt

                            #self.cmd_msg_angular_z=p1
                            #self.cmd_msg_angular_z=p1+self.integrals[0]
                            self.cmd_msg_angular_z=p1+self.integrals[0]+d1 

                            #Restrict the angular z:
                            if self.cmd_msg_angular_z > 0.8:
                                self.cmd_msg_angular_z = 0.8
                            if self.cmd_msg_angular_z < -0.8:
                                self.cmd_msg_angular_z = -0.8

                            #Store the error
                            self.prev_errors[0]=error1

                        else:
                            self.cmd_msg_angular_z=0.0
                            self.integrals[0]=0.0
                        #Top to bottom
                        if not(self.top_center_body_keypoint[1] >= 340 and self.top_center_body_keypoint[1] <=380):
                            #PID gains
                            Kp2=0.8498
                            Ki2=0.0
                            Kd2=(0.336*9.446)/(1+9.446)

                            error2=(360-self.top_center_body_keypoint[1])/300
                            #P
                            p2=Kp2*error2
                            #I
                            self.integrals[1]+=Ki2*error2*self.dt
                            #D
                            d2=Kd2*(error2-self.prev_errors[1])/self.dt

                            #self.cmd_msg_linear_z=p2
                            #self.cmd_msg_linear_z=p2+self.integrals[1]+d2
                            self.cmd_msg_linear_z=Kp2*error2+Ki2*self.integrals[1]

                            #Store the error
                            self.prev_errors[1]=error2

                            #Override
                            self.cmd_msg_linear_z=0.0
                            
                        else:
                            self.cmd_msg_linear_z=0.0
                            self.integrals[1]=0.0

                        #Back-off if too close
                        self.cmd_msg_linear_x = -0.6
                    else:
                        self.state_value = 1800
                        cv2.putText(self.frame, "Body NOT detected :",(20,630), 2, 1.0, (0,255,0), 2)
                        # self.cmd_msg_linear_x=0.0
                        # self.cmd_msg_linear_z=0.0
                        # self.cmd_msg_angular_z=0.0
                        
                else:
                    self.state_value = 1800
                    cv2.putText(self.frame, "Body NOT detected :",(20,630), 2, 1.0, (0,255,0), 2)
                    #self.cmd_msg_linear_x=0.0
                    #self.cmd_msg_linear_z=0.0
                    #self.cmd_msg_angular_z=0.0   
        else:
            self.state_value = 1800
            cv2.putText(self.frame, "Body NOT detected :",(20,630), 2, 1.0, (0,255,0), 2)
            # self.cmd_msg_linear_x=0.0
            # self.cmd_msg_linear_z=0.0
            # self.cmd_msg_angular_z=0.0

        return self.frame,self.cmd_msg_linear_x, self.cmd_msg_angular_z, self.cmd_msg_linear_z, self.prev_errors, self.integrals,self.state_value

  
    def recognize_template(self,frame,matcher,box,track_id,template_encoding):
        """
        Method to start tracking the detected person in the frame if it is recognized.
        
        Args: frame - the image frame
            matcher - instance of a Matcher class - used to match the template with detected object 
            box - bounding box of detected object
            track_id - track IDs of detected object
            template_encoding - encoding of template image
           
        Returns: track_id - track id of a recognized person
            FACE_DETECTED - True if right face is detected and recognized 
            frame - annotated frame
            angular_speed - angular speed
                
        """
        self.frame=frame
        self.matcher=matcher
        self.box=box
        self.track_id=track_id
        self.template_encoding=template_encoding

        self.cmd_msg_angular_z = 0.2        #Specify the angular speed of drone used for searciing around
        self.cmd_msg_linear_x=0.0           #Specify the line
        x,y,w,h=self.box     #x,y (box center), width, height
        yolo_box=int(x-w/2),int(y-h/2),int(w),int(h)
        track=self.track_history[self.track_id]
        track.append((float(x), float(y)))
        if len(track)>self.MAX_TRACK_LENGTH:
                    track.pop(0)
        #Extract the person from the frame
        self.unknown_image=self.frame[int(y)-int(float(h)/2):int(y)+int(float(h)/2),int(x)-int(float(w)/2):int(x)+int(float(w)/2)]
        #Matching
        self.result=self.matcher.match(self.unknown_image,
                                       self.template_encoding,
                                       self.track_id)
        #Person found
        if self.result is not None:
            self.TRACK_ID, self.FACE_DETECTED, self.cmd_msg_angular_z = self.result
            return self.TRACK_ID, self.FACE_DETECTED, self.frame,self.cmd_msg_angular_z 
        #Person not found
        else:
             self.TRACK_ID=None
             self.FACE_DETECTED=False
             return self.TRACK_ID, self.FACE_DETECTED,self.frame,self.cmd_msg_angular_z


class StoreKeypoints():
    """
    Class used to store and remap the body keypoints from YOLOv8-pose to the classic keypoint names.
    """
    def __init__(self,keypoint_detections,i):
        self.NOSE=keypoint_detections[0].xy[0][0]
        self.LEFT_EYE=keypoint_detections[0].xy[0][1]
        self.RIGHT_EYE=keypoint_detections[0].xy[0][2]
        self.LEFT_EAR=keypoint_detections[0].xy[0][3]
        self.RIGHT_EAR=keypoint_detections[0].xy[0][4]
        self.LEFT_SHOULDER=keypoint_detections[0].xy[0][5]
        self.RIGHT_SHOULDER=keypoint_detections[0].xy[0][6]
        self.LEFT_ELBOW=keypoint_detections[0].xy[0][7]
        self.RIGHT_ELBOW=keypoint_detections[0].xy[0][8]
        self.LEFT_WRIST=keypoint_detections[0].xy[0][9]
        self.RIGHT_WRIST=keypoint_detections[0].xy[0][10]
        self.LEFT_HIP=keypoint_detections[0].xy[0][11]
        self.RIGHT_HIP=keypoint_detections[0].xy[0][12]
        self.LEFT_KNEE=keypoint_detections[0].xy[0][13]
        self.RIGHT_KNEE=keypoint_detections[0].xy[0][14]
        self.LEFT_ANKLE=keypoint_detections[0].xy[0][15]
        self.RIGHT_ANKLE=keypoint_detections[0].xy[0][16]
