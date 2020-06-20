# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.
## How to Run
    * Terminal 1:
        cd webservice/server/node-server
        node ./server.js
    * Terminal 2:
       cd webservice/ui
        npm run dev       
    * Terminal 3:
        sudo ffserver -f ./ffmpeg/server.conf
    * Terminal 4:
        python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
        
## Explaining Custom Layers

* Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

## The process behind converting custom layers involves

* Register the custom layer as a extensions to the model optimizer
    For caffe, user has 2 options - 
        1) Register the custom layers as extensions to the Model Optimizer
        2) Register the custom layers as Custom and use the system Caffe to calculate the output shape of each Custom Layer
    For TensorFLow, user has 3 options
        1) Register those layers as extensions to the Model Optimizer.
        2) If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option
        3) Experimental feature of registering definite sub-graphs of the model as those that should be offloaded to TensorFlow during inference
    For MXNet, models with custom layers has again 2 options
        1) Register the custom layers as extensions to the Model Optimizer.
        2) If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option
- Hence, below the process
    * Generate the Extension Template Files Using the Model Extension Generator
    * Using Model Optimizer to Generate IR Files Containing the Custom Layer
    * Edit the CPU Extension Template Files
    * Execute the Model with the Custom Layer
        
## Some of the potential reasons for handling custom layers are 

* To offload computations to TensorFlow:
    Model Optimizer cannot generate an Intermediate Representation from unsupported TensorFlow* operations. However, you can still successfully create an Intermediate Representation if you offload the unsupported operations to TensorFlow for computation. To support this scenario, you must build Inference Engine custom layer with the TensorFlow C++ runtime
    
## Comparing Model Performance

* Here in this project, I used 2 different openvino tensorflow models are used to convert that to intermediate representation and performed the model comparison

## My method(s) to compare models before and after conversion to Intermediate Representations were...
* 1) size of the model
* 2) inference time

## The difference between model accuracy pre- and post-conversion was...

* TensorFlow Object Detection Model Zoo contains many pre-trained models on the coco dataset.
* For this project in the workspace, various classes of models were tested from the TensorFlow Object Detection Model Zoo like SSD MobileNet V2 COCO,  faster_rcnn_inception_v2_coco and faster_rcnn_resnet50_coco. Out of all SSD MobileNet V2 COCO performed good as compared to rest of the models, but, in this project, user can run all the three models with different args pointing to correct model 

SSD MobileNet V2 COCO	 - ssd_mobilenet_v2_coco_2018_03_29.tar.gz

* Model-1: ssd_mobilenet_v2_coco_2018_03_29
    - source http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
        
    - Converted the model to intermediate representation using the below command. 
        - python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

    - The conversion is sucessful, you can find the .xml file and .bin file in workspace/ssd folder . The Execution Time is about 66.47 seconds.



* Model-2: Faster_rcnn_inception_v2_coco_2018_01_28

    - source
    - http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    - Faster R-CNN Inception V2 COCO	 - faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
     
    - Converted the model to intermediate representation using the following command. 
        - python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

    - The conversion is succesful, you can find the .xml and .bin files in the workspace/faster folder. Total execution 154.17 seconds
    - This Inference engine doesnot support dynamic image size so the IR file is generted with the input image size of a fixed size



* Model-3: faster_rcnn_resnet50_coco_2018_01_28

- source
    - http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
    - faster_rcnn_resnet50_coco	 - faster_rcnn_resnet50_coco_2018_01_28.tar.gz
   
* Converted the model to intermediate representation using the following command. 
    - python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config faster_rcnn_resnet50_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

    - The conversion is succesful, you can find the .xml and .bin files in the workspace/faster folder. Total execution 151.29 seconds
    
* Run command to get the performance and inference time
    - python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster2/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
        

## The size of the model pre- and post-conversion was...
## The inference time of the model pre- and post-conversion was...
* ssd_mobilenet_v2_coco - 
    size:
        pre-conversion: 69 MB (size of .pb file)
        post-conversion: 67 MB (size of .bin file)
* faster_rcnn_resnet50_coco_2018_01_28
    size:
        pre-conversion:  120.5 MB (size of .pb file)
        post-conversion: 116.6 MB (size of .bin file)
* Faster_rcnn_inception_v2_coco_2018_01_28
    size:
        pre-conversion:  57.15 MB (size of .pb file)
        post-conversion: 53.2 MB (size of .bin file)
        
* ssd_mobilenet_v2_coco
    inference time: 67 ms
* faster_rcnn_resnet50_coco_2018_01_28
    inference time: 800 ms 
* Faster_rcnn_inception_v2_coco_2018_01_28
    inference time: 880 ms

Comparison
* Comparing the above 3 models i.e. ssd_mobilenet_v2_coco is faster and there is a trade off between latency and memory, It could be clearly seen that the Latency (microseconds) and Memory (Mb) are completly different . I observe the faster_rcnn_resnet size occupies less MB compared to ssd_mobilenet. But the speed wise ssd_mobilenet do it fast.


* Differences in Edge and Cloud computing
* Edge Computing is regarded as ideal for real time operations where latency is a concerns Also, for a minimal task with less data the companies need not to invest more budget for network and cloud infrastructure. hence it will be cost effective. Cloud Computing is more suitable for projects and organizations which deal with massive data storage .


## Assess Model Use Cases

* Following are the different use-cases
    1) Useful to check number of people in a shop/Resturant/Public Park etc.. and can take an action based on the capacity. 
    2) Adding with Tensorflow lite, we can differnetiate the classes of the object. So, can be used to detect the known people and unknow people. Eventualy can be used for security alert system
    3) Also can be used to detect the facemask to allow the person inside/outsie of an organisation/college/private property etc.

Each of these use cases would be useful because...

## Assess Effects on End User Needs

* Need to consider variuos factors like 
    - low Light conditions ,
    - What is the end-user environment conditions like noisy, Thermal etc..
    - How much model accuracy vs perfromance is required for the end user , 
    - What is the end user required Image quality and Image resolution etc..
    
* These above all factors effects on a deployed edge model. So to meet the end-user needs the deployed edge application has to be tested in various scenarios to meet the above conditions and can check the performance in different hardware to have the solution cost-effective. Becuase some edge-applications donot need to have more power consumption and should have low noise environment like medical devices in that case customer may not want a fan based solutions etc. Hence testing model accuracy vs performance on different hardware will allow in determining the best model for the given scenario.

## Model Research

Model 1: ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03
  - [Model Source]
      * I am using the below frozen model from tensor flow
      http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  - I converted the model to an Intermediate Representation with the following arguments...
      * python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb  --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
      
      https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb for images
      * for videos - https://medium.com/@kalpa.subbaiah/easy-video-object-detection-process-using-tensorflow-1e79cb55f94f
      
Model 2: 

- wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz 

* cd /home/workspace
* python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model resources/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config resources/faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

* success - /home/workspace/frozen_inference_graph.xml
* success - /home/workspace/frozen_inference_graph.bin




## [This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

* models - 

Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
