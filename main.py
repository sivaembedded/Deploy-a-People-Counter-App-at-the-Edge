"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import sys
import time
import socket
import json
import cv2
import math

import logging as log
import paho.mqtt.client as mqtt
import numpy as np

from argparse import ArgumentParser
from inference import Network
from inference import Network_ssd
# MQTT server environment variables
Host = socket.gethostname()
IP = socket.gethostbyname(Host)
M_HOST = IP
M_PORT = 3001
M_Time = 60
def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.55,
                        help="Probability threshold for detections filtering"
                        "(0.55 by default)")
    return parser
def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    m_client = mqtt.Client()
    m_client.connect(M_HOST, M_PORT, M_Time)
    return m_client

def draw_outputs(coords, frame, initial_w, initial_h, x, k):
        # Draw output
        # print('Draw Output...')
        current_count = 0     
        ed = x
        for obj in coords[0][0]:
            # Draw bounding box for object when it's probability is more than the specified threshold
            if obj[2] > prob_threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                current_count = current_count + 1
                #print(current_count)
                
                c_x = frame.shape[1]/2
                c_y = frame.shape[0]/2    
                mid_x = (xmax + xmin)/2
                mid_y = (ymax + ymin)/2
                
                # Calculating distance 
                ed =  math.sqrt(math.pow(mid_x - c_x, 2) +  math.pow(mid_y - c_y, 2) * 1.0) 
                k = 0

        if current_count < 1:
            k += 1
            
        if ed>0 and k < 10:
            current_count = 1 
            k += 1 
            if k > 100:
                k = 0
                
        return frame, current_count, ed, k
def infer_on_stream(args, m_client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    global initial_w, initial_h, prob_threshold
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    Model = args.model
    
    Device = args.device
    Cpu = args.cpu_extension
    
    start_time = 0
    cur_request_id = 0
    last_count = 0
    total_count = 0
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(Model, Cpu, Device)
    network_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Checks for live feed
    if args.input == 'CAM':
        input_validated = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_validated = args.input

    # Checks for video file
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(input_validated)
    cap.open(input_validated)
    prob_threshold = args.prob_threshold
    w = int(cap.get(3))
    h = int(cap.get(4))
    temp = 0
    tk = 0
    in_shape = network_shape['image_tensor']

    #iniatilize variables
    
    duration_prev = 0
    counter_total = 0
    dur = 0
    request_id=0
    
    report = 0
    counter = 0
    counter_prev = 0
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (in_shape[3], in_shape[2]))
        image_p = image.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)
  

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': image_p,'image_info': image_p.shape[1:]}
        duration_report = None
        inf_start = time.time()
        infer_network.exec_net(net_input, request_id)

        color = (255,0,0)
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            det_time = time.time() - inf_start
            ### TODO: Get the results of the inference request ###
            net_output = infer_network.get_output()

            # Draw Bounting Box
            frame, current_count, d, tk = draw_outputs(net_output, frame, initial_w, initial_h, temp, tk)

            # Printing Inference Time 
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
 
            # Calculate and send relevant information 
            if current_count > last_count: # New entry
                start_time = time.time()
                total_count = total_count + current_count - last_count
                m_client.publish("person", json.dumps({"total": total_count}))            
            
            if current_count < last_count: # Average Time
                duration = int(time.time() - start_time) 
                m_client.publish("person/duration", json.dumps({"duration": duration}))
           
            # Adding overlays to the frame            
            txt2 = "Distance: %d" %d + " Lost frame: %d" %tk
            cv2.putText(frame, txt2, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
            
            txt2 = "Current count: %d " %current_count
            cv2.putText(frame, txt2, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

            if current_count > 3:
                txt2 = "Alert! Maximum count reached"
                (text_width, text_height) = cv2.getTextSize(txt2, cv2.FONT_HERSHEY_COMPLEX, 0.5, thickness=1)[0]
                text_offset_x = 10
                text_offset_y = frame.shape[0] - 10
                # make the coords of the box with a small padding of two pixels
                box_coords = ((text_offset_x, text_offset_y + 2), (text_offset_x + text_width, text_offset_y - text_height - 2))
                cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, txt2, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)


            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            m_client.publish("person", json.dumps({"count": current_count})) # People Count
            last_count = current_count
            temp = d
        ### TODO: Send the frame to the FFMPEG server ###
        #  Resize the frame
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()
	
def infer_on_stream_ssd(args, client):
    # Initialise the class
    infer_network = Network_ssd()
    # Set Probability threshold for detections
    global initial_w, initial_h, prob_threshold
    Model=args.model
    video_file=args.input    
    extn=args.cpu_extension
    Device=args.device
    #prob_threshold = args.prob_threshold
    
    # Flag for the input image
    single_img_flag = False

    start_time = 0
    cur_request_id = 0
    last_count = 0
    total_count = 0
    
    # Load the model through `infer_network` 
    n, c, h, w = infer_network.load_model(Model, Device, 1, 1, cur_request_id, extn)[1]

    # Handle the input stream
    if video_file == 'CAM': # Check for live feed
        input_stream = 0

    elif video_file.endswith('.jpg') or video_file.endswith('.bmp') :    # Check for input image
        single_img_flag = True
        input_stream = video_file

    else:     # Check for video file
        input_stream = video_file
        assert os.path.isfile(video_file), "Specified input file doesn't exist"
    
    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
        
    total_count = 0  
    duration = 0
    
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    prob_threshold = args.prob_threshold
    temp = 0
    tk = 0
    
    # Loop until stream is over
    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        # Pre-process the image as needed
        # Start async inference
        image = cv2.resize(frame, (w, h))
        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        # Start asynchronous inference for specified request
        inf_start = time.time()
        infer_network.exec_net(cur_request_id, image)
        
        color = (255,0,0)

        # Wait for the result
        if infer_network.wait(cur_request_id) == 0:
            det_time = time.time() - inf_start

            # Get the results of the inference request 
            result = infer_network.get_output(cur_request_id)
            
            # Draw Bounting Box
            frame, current_count, d, tk = draw_outputs(result, frame, initial_w, initial_h, temp, tk)
            
            # Printing Inference Time 
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
            
            # Calculate and send relevant information 
            if current_count > last_count: # New entry
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))            
            
            if current_count < last_count: # Average Time
                duration = int(time.time() - start_time) 
                client.publish("person/duration", json.dumps({"duration": duration}))
           
            # Adding overlays to the frame            
            txt2 = "Distance: %d" %d + " Lost frame: %d" %tk
            cv2.putText(frame, txt2, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
            
            txt2 = "Current count: %d " %current_count
            cv2.putText(frame, txt2, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

            if current_count > 3:
                txt2 = "Alert! Maximum count reached"
                (text_width, text_height) = cv2.getTextSize(txt2, cv2.FONT_HERSHEY_COMPLEX, 0.5, thickness=1)[0]
                text_offset_x = 10
                text_offset_y = frame.shape[0] - 10
                # make the coords of the box with a small padding of two pixels
                box_coords = ((text_offset_x, text_offset_y + 2), (text_offset_x + text_width, text_offset_y - text_height - 2))
                cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
                
                cv2.putText(frame, txt2, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            
            client.publish("person", json.dumps({"count": current_count})) # People Count

            last_count = current_count
            temp = d

            if key_pressed == 27:
                break

        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        
        #Save the Image
        cv2.imwrite('output_image.jpg', frame)
       
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    
    m_client = connect_mqtt()
    # Perform inference on the input stream
    if args.model == 'faster/frozen_inference_graph.xml':
        infer_on_stream(args, m_client)
    elif args.model == 'faster2/frozen_inference_graph.xml':
        infer_on_stream(args, m_client)
    elif args.model == 'ssd/frozen_inference_graph.xml':
        infer_on_stream_ssd(args, m_client)
    
if __name__ == '__main__':
    main()
