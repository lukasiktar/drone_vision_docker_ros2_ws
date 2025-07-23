import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
#from sensor_msgs.msg import Joy
from message_filters import ApproximateTimeSynchronizer, Subscriber

from ultralytics import YOLO
import cv2 

from drone_vision_control import Recognition
from drone_vision_control import Detection
from collections import defaultdict
import logging
import os


#Node:
class DroneNode(Node):

    def __init__(self):
        super().__init__('drone_node')

        # #Define the codec and create VideoWriter object
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        #self.out=cv2.VideoWriter("./drone_flight01.mp4", fourcc,33, (640,480))
        self.out=cv2.VideoWriter("./drone_flight01.mp4", fourcc,10, (1280,720)) #for RealsenseCamera

        #Define used variables
        self.TRACK_ID=None          #The ID of the object to track
        self.FACE_DETECTED=False    #Bool value if the object is detected
        self.FACE_STORED=False      #Bool value if the stored face template exists
        self.prev_time=self.get_clock().now().nanoseconds/ 1e9  #Get starting time
        self.state_value=1495       #The dafault state value for drone (free drive)
        self.PERSON_FOLLOW_STOPPED=False #The flag that raises when the following stopped
        
        #Previous distance error values (for the derivative effect)
        self.prev_error1,self.prev_error2,self.prev_error3,self.prev_error4=0,0,0,0
        self.prev_errors=[self.prev_error1, self.prev_error2, self.prev_error3, self.prev_error4]

        #Integral values (for the integral effect)
        self.integral1,self.integral2,self.integral3,self.integral4=0,0,0,0
        self.integrals=[self.integral1, self.integral2, self.integral3, self.integral4]

        #Template image if exists
        template_image_path="template_images/LukaS.jpg"
        #Store the person name
        file_name=os.path.splitext(template_image_path)[0]
        self.name=file_name.split('/')[-1]
        print(f"Searched person: {self.name}")

        #Template encoding
        template=Recognition.Recognition()
        self.template_encoding=template.encode_template(template_image_path=template_image_path)

        #Track history
        self.track_history=defaultdict(lambda:[])

        #Define matcher
        self.matcher=Recognition.Matcher()
        
        #Turn off logger (yolo prints the detection log by default)
        logging.getLogger('ultralytics').setLevel(logging.WARNING)

        #Tracker setup 
        self.tracker=Detection.Detection(matcher=self.matcher)
        self.model=self.tracker.setup_detector('yolov8n.pt', ["person"])
        #self.model=self.tracker.setup_detector('yolov11n.pt', ["person"])

        #YOLO Pose model
        self.pose_model = YOLO("yolov8n-pose.pt")
        #self.pose_model = YOLO("yolo11n-pose.pt")

        #Setup the ROS image publisher
        self.publisher_ = self.create_publisher(Image, "processed_image" , 10)
        #Setup the ROS drone velocity publisher (for "Free drive")
        self.cmd_vel_pub2 = self.create_publisher(Twist, "/drone_cmd_vel", 10)
        #Setup the ROS state value publisher (for drone state and drone colors)
        self.state_publisher=self.create_publisher(Int32,"/state_value",10)

        self.br = CvBridge()
        self.cmd_msg = Twist()

        # Subscribers
        self.rgb_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        # Synchronize the topics outputs
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.img_callback)
        
        #Ardupilot commands
        self.start_search_srv = self.create_service(Trigger, 'search', self.start_search_callback)
        self.stop_follow_srv = self.create_service(Trigger, 'stop', self.stop_follow_callback)
        self.start_follow_srv = self.create_service(Trigger, 'follow', self.start_follow_callback)

        #Additional functions (no working for now)
        self.start_left_circle_srv = self.create_service(Trigger, 'b_left', self.start_left_circle_callback)
        self.start_right_circle_srv = self.create_service(Trigger, 'b_right', self.start_right_circle_callback)
        self.increase_radius_srv = self.create_service(Trigger, 'b_up', self.start_increase_radius_callback)
        self.decrease_radius_srv = self.create_service(Trigger, 'b_down', self.start_decrease_radius_callback)

        #Initial drone state
        self.state = "free"         #State of flight
        self.previous_state="free"  #Previous state of flight (used to monitor the commands flow)
        self.radius_state=400  #280 cm is the starting distance between drone and person
                
    def match_check():
        return True
    
    #Callback functions for setting desired state:
    def start_follow_callback(self, request, response):
        self.get_logger().info('Received start_follow request')
        if self.state != 'follow':
            self.previous_state=self.state
            self.state='follow'
            #Starting radius state
            self.radius_state=400
            # Follow state int message for drone
            self.state_value = 1800  

            #Reset the errors and integrals for PID regulator
            #Previous distance error values (for the derivative effect)
            self.prev_error1,self.prev_error2,self.prev_error3,self.prev_error4=0,0,0,0
            self.prev_errors=[self.prev_error1, self.prev_error2, self.prev_error3, self.prev_error4]

            #Integral values (for the integral effect)
            self.integral1,self.integral2,self.integral3,self.integral4=0,0,0,0
            self.integrals=[self.integral1, self.integral2, self.integral3, self.integral4]

        response.success = True
        response.message = 'Follow started successfully'
        return response

    def stop_follow_callback(self, request, response):
        self.get_logger().info('Received stop_follow request')
        if self.state != 'free':
            print("Free")
            self.previous_state=self.state
            self.state='free'
            #Free drive int message for drone
            self.state_value = 1495
        response.success = True
        response.message = 'Follow stopped successfully'
        return response

    def start_search_callback(self, request, response):
        self.get_logger().info('Received start_search request')
        if self.state != 'search':
            self.previous_state=self.state
            self.state='search'
            self.FACE_STORED=False
            #If the template image is stored before, run the search by template
            if self.template_encoding is not None:
                print("search_template")
                self.state='search_template'
                self.FACE_STORED=True       #The face is found and stored
            #Search int message for drone
            self.state_value = 1700

            #Reset the errors and integrals for PID regulator
            #Previous distance error values (for the derivative effect)
            self.prev_error1,self.prev_error2,self.prev_error3,self.prev_error4=0,0,0,0
            self.prev_errors=[self.prev_error1, self.prev_error2, self.prev_error3, self.prev_error4]

            #Integral values (for the integral effect)
            self.integral1,self.integral2,self.integral3,self.integral4=0,0,0,0
            self.integrals=[self.integral1, self.integral2, self.integral3, self.integral4]
            
        response.success = True
        response.message = 'Search started successfully'
        return response

    def start_left_circle_callback(self, request, response):
        self.get_logger().info('Received start_left_circle request')
        if self.state != 'left_circle':
            self.previous_state=self.state
            self.state='left_circle'
        response.success = True
        response.message = 'Left circle started successfully'
        return response

    def start_right_circle_callback(self, request, response):
        self.get_logger().info('Received start_right_circle request')
        if self.state != 'right_circle':
            self.previous_state=self.state
            self.state='right_circle'
        response.success = True
        response.message = 'Right circle started successfully'
        return response
    
    def start_increase_radius_callback(self, request, response):
        self.get_logger().info('Received increase radius request')
        self.radius_state += 1
        response.success = True
        response.message = 'Radius increased successfully'
        return response
    
    def start_decrease_radius_callback(self, request, response):
        self.get_logger().info('Received decrease radius request')
        self.radius_state -= 1
        response.success = True
        response.message = 'Radius decreased successfully'
        return response
   
    
    #Image callback
    def img_callback(self, rgb_msg, depth_msg):
        rgb_image = self.br.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')     #Read RGB image 
        self.depth_image = self.br.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')     #Read depth image

        #Resize the depth image to rgb image format 
        self.depth_image = cv2.resize(self.depth_image, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        #Change the rgb_image to the RGB
        frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        #Get the tracking results
        results=self.tracker.track(frame)
        cv2.putText(frame, "CRTA DRONE TRACKING ",(450,30), 2, 1.0, (255,255,255), 1)  #Title 
        #Crosshair in the middle
        cv2.line(frame, (int(frame.shape[1]/2)-20,int(frame.shape[0]/2)), (int(frame.shape[1]/2)+20,int(frame.shape[0]/2)),(255,255,255),2)
        cv2.line(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)-20), (int(frame.shape[1]/2),int(frame.shape[0]/2)+20),(255,255,255),2)

        #Get the boxes and track IDs of detected individuals
        if results.boxes.id!= None:
            #Parse the boxes and track_id-s
            boxes=results.boxes.xywh.cpu()
            track_ids=results.boxes.id.int().cpu().tolist()
            
            if self.state=='free':
                #Free drive
                self.FACE_DETECTED=False        #Reset the face search - there is no detected faces
                self.TRACK_ID=None              #Reset the face search - there is no detected ids
                #Reset the speeds in case of remaining speeds
                self.cmd_msg.linear.x=0.0
                self.cmd_msg.linear.z=0.0 
                self.cmd_msg.angular.z = 0.0
                self.cmd_msg.linear.y=0.0
                self.PERSON_FOLLOW_STOPPED=False    #Flag for stopping person following
                cv2.putText(frame, "MODE", (20,360), 2, 1.0, (200,200,200),2)
                cv2.putText(frame, "Free drive",(20,400), 2, 1.0, (255,255,255), 2)
                cv2.putText(frame, "People detected",(20,700), 2, 1.0, (255,255,255), 2)

            elif self.state=='search':
                #Search mode without face template
                box=None
                track_id=None
                #If the face is aimed - we know the TRACK_ID and can exclude searching repeatedly
                if self.FACE_DETECTED==True:
                    for box, track_id in zip(boxes, track_ids):
                        if track_id==self.TRACK_ID:
                            self.cmd_msg.angular.z = 0.0
                            self.cmd_msg.linear.x=0.0
                            self.cmd_msg.linear.z=0.0
                            self.cmd_msg.linear.y=0.0

                            # Calculate time
                            current_time = self.get_clock().now().nanoseconds / 1e9  # Get current time
                            dt = current_time - self.prev_time  # Calculate time difference
                            self.prev_time = current_time  # Update prev_time for next iteration

                            cv2.putText(frame, "MODE:", (20,360),2,1.0, (200,200,200),2)
                            cv2.putText(frame, "Search mode",(20,400), 2, 1.0, (255,255,255), 2)
                           
                            #Center the body
                            frame, self.cmd_msg.linear.x, self.cmd_msg.angular.z, self.cmd_msg.linear.z,self.prev_errors,self.integrals=self.tracker.center_detected_body(box,
                                                                                                                                                                        track_id,
                                                                                                                                                                        self.pose_model,
                                                                                                                                                                        self.prev_errors,
                                                                                                                                                                        self.integrals,
                                                                                                                                                                        dt,
                                                                                                                                                                        self.name,  
                                                                                                                                                                        self.state_value)                       
                else:
                    #Check if the drone aims to the person - to start centering
                    for box0, track_id0 in zip(boxes, track_ids):
                        if box0[0]-box0[2]/2-50 < int(frame.shape[1]/2) and box0[0]+box0[2]/2+50> int(frame.shape[1]/2) and box0[1]-box0[3]/2-50< int(frame.shape[0]/2) and box0[1]+box0[3]/2+50> int(frame.shape[0]/2):
                            box=box0
                            track_id=track_id0
                            self.TRACK_ID=track_id
                    self.name="Unknown" #The found person is unknown

                    if box is not None:
                        #In case no faces are found and stored 
                        if self.FACE_STORED==False:
                            #Save the image of the found person as template
                            cv2.imwrite("template_image.jpg",frame)
                            cv2.putText(frame, "Face stored",(20,500), 2, 1.0, (255,255,255), 2)

                            #Calculate the embedding of the template
                            template_image_path="template_image.jpg"
                            template=Recognition.Recognition()
                            self.template_encoding=template.encode_template(template_image_path=template_image_path)

                            self.FACE_DETECTED=True     #The face is detected - flag that enables tracking
                            self.FACE_STORED=True       #The face is found and stored - flag that enables search by template
                            
                        # Calculate time
                        current_time = self.get_clock().now().nanoseconds / 1e9     # Get current time
                        dt = current_time - self.prev_time          # Calculate time difference
                        self.prev_time = current_time       # Update prev_time for next iteration

                        cv2.putText(frame, "MODE:",(20,360), 2, 1.0, (200,200,200), 2)
                        cv2.putText(frame, "Search mode",(20,400), 2, 1.0, (255,255,255), 2)
                        
                        self.cmd_msg.linear.y=0.0 #Reset the speed in y direction

                        #Search successful int message to drone
                        self.state_value = 1750

                        #Center the body in the image
                        frame,self.cmd_msg.linear.x, self.cmd_msg.angular.z, self.cmd_msg.linear.z,self.prev_errors,self.integrals,self.state_value=self.tracker.center_detected_body(box,
                                                                                                                                                                                    track_id,
                                                                                                                                                                                    self.depth_image,
                                                                                                                                                                                    self.pose_model,
                                                                                                                                                                                    self.prev_errors,
                                                                                                                                                                                    self.integrals,
                                                                                                                                                                                    dt,
                                                                                                                                                                                    self.name,
                                                                                                                                                                                    self.state_value,
                                                                                                                                                                                    )                       
                    
                    else:
                        #In a case the person disappeared from image
                        self.state_value = 1700
                        cv2.putText(frame, "Search mode stopped",(20,500), 2, 1.0, (255,255,255), 2)
                        #Reset the speeds
                        self.cmd_msg.linear.x=0.0
                        self.cmd_msg.angular.z=0.0
                        self.cmd_msg.linear.z=0.0
                        
                        

            elif self.state=="follow" and (self.previous_state=="search" or self.previous_state=="search_template"):
                #Follow mode (activated only if face is found and centered before)
                #Loop through all detections and extact only the right one
                for box, track_id in zip(boxes, track_ids):
                    cv2.putText(frame, "MODE:",(20,360), 2, 1.0, (200,200,200), 2)
                    cv2.putText(frame, "Follow mode",(20,400), 2, 1.0, (255,255,255), 2)

                    if self.FACE_DETECTED==True and track_id==self.TRACK_ID:
                        # Calculate time
                        current_time = self.get_clock().now().nanoseconds / 1e9  # Get current time
                        dt = current_time - self.prev_time  # Calculate time difference
                        self.prev_time = current_time  # Update prev_time for next iteration

                        self.cmd_msg.linear.y=0.0   #Reset the speed in y direction
                        self.cmd_msg.linear.x=0.0
                        self.cmd_msg.angular.z=0.0
                        self.cmd_msg.linear.z=0.0
                        #Follow successful int message for drone
                        self.state_value = 1850

                        #Track the body
                        frame, self.cmd_msg.linear.x, self.cmd_msg.angular.z, self.cmd_msg.linear.z, self.prev_errors, self.integrals,self.state_value=self.tracker.track_detected_body(box,
                                                                                                                                                                                        track_id,
                                                                                                                                                                                        self.depth_image,
                                                                                                                                                                                        self.pose_model,
                                                                                                                                                                                        self.prev_errors, 
                                                                                                                                                                                        self.integrals,
                                                                                                                                                                                        dt,
                                                                                                                                                                                        self.name, 
                                                                                                                                                                                        self.radius_state,
                                                                                                                                                                                        self.state_value,
                                                                                                                                                                                        )
                        
                        self.PERSON_FOLLOW_STOPPED=False #If the person follow did not stopped
                        continue
                    else:
                        #The person disappeared from image
                        self.state_value = 1800
                        cv2.putText(frame, "Follow mode stopped",(20,500), 2, 1.0, (255,255,255), 2)
                        #Reset the speeds
                        self.cmd_msg.linear.x=0.0
                        self.cmd_msg.angular.z=0.0
                        self.cmd_msg.linear.z=0.0

                        # TRACK FAIL FIX
                        #If person probably did not disappeared but does not have the ID, try to find it again
                        self.previous_state=self.state
                        self.state='search_template'
                        self.FACE_STORED=True
                        self.state_value = 1700

                        #TRACK FAIL FIX
                    
                        

            elif self.state=="left_circle" and (self.previous_state=="search" or self.previous_state=="search_template"):
                #Perform left circle aronud the specified individual
                #Loop through all detections and extact only the right one
                for box, track_id in zip(boxes, track_ids):
                    cv2.putText(frame, "MODE:",(20,360), 2, 1.0, (200,200,200), 2)
                    cv2.putText(frame, "Orbit mode (left)",(20,400), 2, 1.0, (255,255,255), 2)
                    self.circle_direction="LEFT"

                    if self.FACE_DETECTED==True and track_id==self.TRACK_ID:
                        # Calculate time
                        current_time = self.get_clock().now().nanoseconds / 1e9  # Get current time
                        dt = current_time - self.prev_time  # Calculate time difference
                        self.prev_time = current_time  # Update prev_time for next iteration
                    
                        #Track the body
                        frame, self.cmd_msg.linear.x, self.cmd_msg.angular.z, self.cmd_msg.linear.z, self.prev_errors, self.integrals,self.state_value=self.tracker.track_detected_body(box,
                                                                                                                                                                                        track_id,
                                                                                                                                                                                        self.depth_image,
                                                                                                                                                                                        self.pose_model,
                                                                                                                                                                                        self.prev_errors, 
                                                                                                                                                                                        self.integrals,
                                                                                                                                                                                        dt,
                                                                                                                                                                                        self.name, 
                                                                                                                                                                                        self.radius_state,
                                                                                                                                                                                        self.state_value,
                                                                                                                                                                                        )
                        self.PERSON_FOLLOW_STOPPED=False #If the person follow did not stopped
                        self.cmd_msg.linear.y=0.2
                    else:
                        self.cmd_msg.linear.y=0.0

            elif self.state=="right_circle" and (self.previous_state=="search" or self.previous_state=="search_template"):
                #Loop through all detections and extact only the right one
                for box, track_id in zip(boxes, track_ids):
                    cv2.putText(frame, "MODE:",(20,360), 2, 1.0, (200,200,200), 2)
                    cv2.putText(frame, "Orbit mode (right)",(20,400), 2, 1.0, (255,255,255), 2)
                    self.circle_direction="RIGHT"

                    if self.FACE_DETECTED==True and track_id==self.TRACK_ID:
                        # Calculate time
                        current_time = self.get_clock().now().nanoseconds / 1e9  # Get current time
                        dt = current_time - self.prev_time  # Calculate time difference
                        self.prev_time = current_time  # Update prev_time for next iteration

                        #Track the body
                        frame, self.cmd_msg.linear.x, self.cmd_msg.angular.z, self.cmd_msg.linear.z, self.prev_errors, self.integrals,self.state_value=self.tracker.track_detected_body(box,
                                                                                                                                                                                        track_id,
                                                                                                                                                                                        self.depth_image,
                                                                                                                                                                                        self.pose_model,
                                                                                                                                                                                        self.prev_errors, 
                                                                                                                                                                                        self.integrals,
                                                                                                                                                                                        dt,
                                                                                                                                                                                        self.name, 
                                                                                                                                                                                        self.radius_state,
                                                                                                                                                                                        self.state_value,
                                                                                                                                                                                        )
                        self.PERSON_FOLLOW_STOPPED=False #If the person follow did not stopped
                        self.cmd_msg.linear.y=-0.2
                    else:
                        self.cmd_msg.linear.y=0.0
            

            elif self.state=='search_template' and self.FACE_STORED==True:
                cv2.putText(frame, "MODE:",(20,360), 2, 1.0, (200,200,200), 2)
                cv2.putText(frame, "Search Template",(20,400), 2, 1.0, (255,255,255), 2)

                #Loop through all detections and extact only the right one
                for box, track_id in zip(boxes, track_ids):

                    #Proceed only for specified ID that represents recognized person
                    if self.FACE_DETECTED==True:
                        if track_id==self.TRACK_ID:
                            self.cmd_msg.angular.z = 0.0
                            self.cmd_msg.linear.x=0.0
                            self.cmd_msg.linear.z=0.0
                            self.cmd_msg.linear.y=0.0

                            # Calculate time
                            current_time = self.get_clock().now().nanoseconds / 1e9  # Get current time
                            dt = current_time - self.prev_time  # Calculate time difference
                            self.prev_time = current_time  # Update prev_time for next iteration

                            self.state_value = 1750
                            #Center the body in the image
                            frame,self.cmd_msg.linear.x, self.cmd_msg.angular.z, self.cmd_msg.linear.z,self.prev_errors,self.integrals,self.state_value=self.tracker.center_detected_body(box,
                                                                                                                                                                                        track_id,
                                                                                                                                                                                        self.depth_image,
                                                                                                                                                                                        self.pose_model,
                                                                                                                                                                                        self.prev_errors,
                                                                                                                                                                                        self.integrals,
                                                                                                                                                                                        dt,
                                                                                                                                                                                        self.name,
                                                                                                                                                                                        self.state_value,
                                                                                                                                                                                        ) 
                            continue
                    #Proceed if there is a template but the person is not found yet
                    elif self.FACE_DETECTED==False:
                        self.state_value = 1700
                        cv2.putText(frame, "People detected",(20,700), 2, 1.0, (255,255,255), 2)
                        self.cmd_msg.angular.z = 0.2
                        self.cmd_msg.linear.x=0.0
                        self.cmd_msg.linear.z=0.0
                        self.cmd_msg.linear.y=0.0
                        #Match the template with the detected face
                        self.TRACK_ID,self.FACE_DETECTED,frame, self.cmd_msg.angular.z=self.tracker.recognize_template(frame,
                                                                                                                        self.matcher,
                                                                                                                        box,track_id,
                                                                                                                        self.template_encoding)
                        #If the person was followed and the person is detected in this step then start following
                        if self.PERSON_FOLLOW_STOPPED==True and self.FACE_DETECTED==True:
                            self.state="follow"
                            self.state_value = 1800
                   
            
        else:
            #If the person was followed and then lost, start the search template process
            if self.state == "follow":
                self.PERSON_FOLLOW_STOPPED=True #Follow stopped
                self.state="search_template"    #Start search template
            self.FACE_DETECTED=False        #Reset the flag
            self.TRACK_ID=None              #Reset the flag
            self.cmd_msg.linear.x=0.0       #Reset the speed
            self.cmd_msg.linear.z=0.0       #Reset the speed
            self.state_value = 1495         #Set the state value
            
            #If the search_template state is started set the rotation speed
            if self.state == "search_template": 
                cv2.putText(frame, "MODE:", (20,360),2,1.0, (200,200,200),2)
                cv2.putText(frame, "Search Template",(20,400), 2, 1.0, (255,255,255), 2)
                self.cmd_msg.angular.z = 0.2
            else:
                self.cmd_msg.angular.z = 0.0   #Reset the speed
                
            #If the person was followed and then lost, start the search template process
            if self.state == "follow":
                self.PERSON_FOLLOW_STOPPED=True     #Follow stopped
                self.state = "search_template"      #Start search template
            if self.state == "free":
                cv2.putText(frame, "MODE:", (20,360),2,1.0, (200,200,200),2)
                cv2.putText(frame, "Free drive",(20,400), 2, 1.0, (255,255,255), 2)
                
        #Publish the data:     
        self.cmd_vel_pub2.publish(self.cmd_msg)     #Control speeds sent to the drone 
        #Processed frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.publisher_.publish(self.br.cv2_to_imgmsg(frame))
        #State message
        msg=Int32()
        msg.data=self.state_value
        self.state_publisher.publish(msg)
        #Store the video frame
        self.out.write(frame)
            

def main(args=None):
    #ROS initialization
    rclpy.init(args=args)
    #Creation of a node
    drone_node = DroneNode()
    rclpy.spin(drone_node)
    #Destory the node
    drone_node.destroy_node()
    #ROS shutdown
    rclpy.shutdown()

  
if __name__ == '__main__':
  main()
