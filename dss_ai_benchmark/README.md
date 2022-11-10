# dss_ai_benchmark

A custom benchmark tool to evaluate read access of DSS Object storage through deep neural networks such as PyTorch , TensorFlow.
The tool gives the flexibility to the user to add their custom dataset, training and neural network based model.These classes can be written into the respective files and can be configured through configuration file.

As per the latest version of the tool, we have 2 projects live -- an Image Classification project; and an Object Detection project. User can choose to run any one of them at a time. The tool is fairly automated to run the project of the user's choice, provided some modifications to the "choice" attributes are made under each header within the **Config** file -- this has been discussed in the ***section #3*** below...

## 1. Project List

- This tool initially was created with an Image Classification framework where user is supposed to provide a dataset for a few categories, train the model on that dataset and optional inference.
- But now it has been expanded to include an Object Detection framework along with the existing Image Classification framework. This has been created keeping in mind the usage of a production-level framework to test out our performance metrics on! User needs to come up with 3-5 categories of objects, with their corresponding image dataset (containing more than at least 7000 images), along with their bounding-box coordinates (coordinates (x_min, y_min, x_max, y_max) on the image to tell the model where to locate the specific object), and feed them in a specific format to the tool -- which will then train and return a trained model which can be further used for inference.

## 2. Setup and Installation

Install the required libraries by running the following command in the node you are working on.

```bash
python3 -m pip install -r requirements.txt
```

To work with dss_client lib, one require to install that library separately on the node.

## 3. Configuration

The custom usage requires user to update dataset, model and training in the **Config** file to activate either of the projects, as required. User needs to update the following sections:

- Set the Deep Learning model Framework. (PyTorch or TensorFlow; but currently, only PyTorch is supported)

  ```json
  {
  "framework": {"name": "PyTorch",   <-- Update the name here
                "TensorFlow": {
  
                },
                "PyTorch": {
                  "DataLoader": {
                    "shuffle": true,
                    .
                    .
                    ...
                  }
                }
            }
  }
  ```

- Update the Model to be used corresponding to the specific project ("ConvNet" for ImageClassification or "ObjectDetector for ObjectDetection)

  ```json
  "model": {
    "choice": "ConvNet",      <-- Update the choice here

    "ConvNet": {
        "loss_function": "<Specify Loss Function ex.[CorssEntrophy]"
    },
    "ObjectDetector": {
        "loss_function": ["MSE", "CrossEntropy"]
    }
  }
  ```

- Update the Training class name corresponding to the project to be run ("RandomAccessDatasetTrain" for Image Classification or "ObjectDetectionDatasetTrain" for Object Detection)

  ```json
  "training": { 
      "choice": "RandomAccessDatasetTrain",     <-- Update the choice here

      "RandomAccessDatasetTrain": {
          "__comment__": "Whether we are going for a Random access or Sequential access",
          "access_method": "Random"
      },
      "ObjectDetectionDatasetTrain": {
          "__comment__": "Specify the Learning Rate, the initial Loss weights",
          "init_lr": 1e-4,
          "label_loss": 1.0,
          "bbox_loss": 1.0
      }
  }
  ```

- Update the Dataset class name corresponding to the project to be run ("TorchImageClassificationDataset" for Image Classification or "TorchObjectDetectionDataset" for Object Detection)

  ```json
  "dataset": {
              "choice": "TorchImageClassificationDataset",   <-- Update the choice here

              "TorchImageClassificationDataset": {
                    "access": {"random": true, "sequential": false},
                    "type": "image",
                    ...
              },
              "TorchObjectDetectionDataset": {
                    "access": {"random": true, "sequential": false},
                    "type": "image",
                    ...
              },
  }
  ```

- Update the "Storage" section for the specific directories and/or files to be used corresponding to the project being run ("ConvNet" for Image Classification or "ObjectDetector" for Object Detection)
  - In case of inputs from S3, update the section related to `s3` below -- like the "client_lib" to be used, the "bucket" name, the corresponding "prefixes" where the dataset is kept, etc (shown below) and also toggle the "format" & "name" attribute value between "fs", "nfs" and "s3", "dss" -- depending on the use case.  

  ```json
  "storage": {
      "format": "s3",             <=== Update format
      "name": "dss",              <=== Update name
      "s3": {
        "bucket": "dss0",               <==== Specify Bucket
        "prefix": ["flower_photos/"],    <==== Specify prefix
        "client_lib": {
          "name": "boto3",               <=== Specify Client Lib
          "dss_client": {"max_connections":25, "http_request_timeout_ms":0, "connect_timeout_ms":1000,
                         "request_timeout_ms": 10000, "enable_tcp_keep_alive":true,
                         "tcp_keep_alive_interval_ms":  1000, "object_keys_per_page_count":  1000,
                         "endpoints_per_cluster": 256}
        },
        "aws": {
          "credentials": {
            "region_name": "us-east-2",
            "access_key": "access_key",
            "secret_key": "secret_key"
          }
        },
        "dss": {
          "credentials": {                             <=== Specify endpoint & credentials
            "endpoint": "http://10.1.51.107:9000",
            "access_key": "minio",
            "secret_key": "minio123"
          }
        }
      },
      "fs": {                                  <=== Update corresponding details here if working with Filesystem inputs
         "choice": "ConvNet",
         "ConvNet": {
           "data_dir": ["/nfs_share1"],
           "base_output_dir": "image_classifier_outputs",
           "saved_model_name": "classifier.pt",
           "predictions_path": "image_classifier_outputs/predicted_images"
         },
         "ObjectDetector": {
           "data_dir": ["/Dss_Datasets/google_open_images/OID/Dataset"],
           "base_output_dir": "object_detector_outputs",
           "saved_model_name": "detector.pth",
           "le_name": "label_encoder.pickle",
           "plots_path": "object_detector_outputs/plots",
           "predictions_path": "object_detector_outputs/predicted_images",
           "test_paths": "object_detector_outputs/test_paths.txt"
         }
      }
  }
  ```

- Update the paths where the logs of the tool and the metrics need to be stored (This is irrespective of the project being run -- as both will have their own corresponding logs and metrics being printed).
  - Saved metrics file name: **metrics.csv**
  - Saved log file name: **benchmark.log**

  ```json
  "metrics": {
      "format": "csv",
      "graph": {"name": "SampleGraph", "enabled": false},
      "path": "/usr/dss/dss_ai_benchmark"    <-- Update metrics collection path here
  },
  "logging": { "path": /usr/dss/dss_ai_benchmark,     <-- Update logging path here
               "level": "INFO"
  }
  ```

## 4. Execution

Once all the "choices" in the **Config** file are updated corresponding to the project you wish to run (as per the ***#3. Configuration*** section provided above), go ahead and run the tool as below:

```bash
# Use configuration file from default location.
python3 benchmark.py

# Use configuration file from custom location
python3 benchmark.py -cfg <configuration fle>

# To run with dss_client lib use the following command
sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 benchmark.py'   <-- Note: as of now dss_client is not being supported.
```

## 5. Workings of the Image Classification project

### 5.1. Add Data source

The image classification project should have data in the following structures.
The multiple data source is supported with same set of categories present into each data source. Let say we want to solve a classification problem of flowers. It requires to provide source data of multiple type of flowers. These are called categories.
Each flower directories should have images. In the following example we have five categories of flower images.

```text
## File System Structure , specify with data_dir as list
# Source1
/flower_photos1/daisy
/flower_photos1/roses
/flower_photos1/dandelion    
/flower_photos1/sunflowers
/flower_photos1/tulips
# Source2
/flower_photos2/daisy
/flower_photos2/roses
/flower_photos2/dandelion    
/flower_photos2/sunflowers
/flower_photos2/tulips

## S3 structure for AWS , specify under same bucket
# Source1
bucket/flower_photos1/daisy
bucket/flower_photos1/roses
bucket/flower_photos1/dandelion    
bucket/flower_photos1/sunflowers
bucket/flower_photos1/tulips
# Source2
bucket/flower_photos2/daisy
bucket/flower_photos2/roses
bucket/flower_photos2/dandelion    
bucket/flower_photos2/sunflowers
bucket/flower_photos2/tulips

## DSS Should hide bucket from user. So, bucket name is not relevant for that 
```

The tool support file system data as well as S3 data. Update configuration accordingly.
For AWS, we are supposed to add bucket name (bucket).
The supported client_lib is "boto3" or "dss_client". The "boto3" should support of type of object storage.
But, "dss_client" only works for stock minio and dss-minio storage.

```json
"storage": {
      "format": "fs",  <<<== Supported formats fs ( File Systems ), s3
      "name": "nfs",  <<<== File System name such as nfs/wekafs or dss/aws for s3
      "s3": {
        "bucket": "bucket",
        "prefix": ["flower_photos/", "flower_photos2/"],
        "client_lib": "boto3", <<<== Supported lib boto3, dss_client
        "aws": {
          "credentials": {
            "region_name": "us-east-2",
            "access_key": "access_key",
            "secret_key": "secret_key"
          }
        },
        "dss": {
          "credentials": {
            "endpoint": "http://202.0.0.137:9000",
            "access_key": "minio",
            "secret_key": "minio123"
          }
        }
      },
      "fs": {
      "choice": "ConvNet", 
      "ConvNet": {"data_dir": ["/data/datasets/flower_photos", "/data/datasets/flower_photos2"]
                 }
      }
```

### 5.2. Create a CustomData class at  the following file

```python
dataset/pytorch_dataset.py
class MyDataset(RandomAccessDataset):
    """
    User defined dataset class. The base class has all default functionalities
    Each functions in the override section can be overide
    """

    def __init__(self, config={}):
        super(MyDataset, self).__init__(transform=None,
                                        config=config)
    def read_file_system_data(self, image):
         """
         Add your code here to read filesystem data
         """
    def read_s3_object(self, image):
        """
        Updated code is require to read data from S3
        """

Example: A sample TorchImageClassificationDataset is provided based on MapStyle or RandomAccess type.
```

The Configuration file is as below.
If using MapStyle/RandomAccess , then set random to true.
Use the image dimension to 32x32 , or update into torch size.
Update categories to read from.

```json
"dataset": {
               "choice": "TorchImageClassificationDataset",

               "TorchImageClassificationDataset": {
                    "access": {"random": true, "sequential": false},
                    "type": "image",
                    "label": ["daisy", "dandelion", "roses", "sunflowers", "tulips"],
                    "image_dimension": [32, 32],

                    "__comment__": "Below line specifies the average image file size in 'KB'; has been included --",
                    "__comment__": "-- to avoid extra file reads to extract the dataset size.",
                    "avg_image_size": 62
               },

## Add / update training class
```

### 5.3. Available dataset class and corresponding training class

"PythonReadDataset" => "PythonReadTrain"   # This skip actual model training, only measure read performance.
"TorchImageClassificationDataset" => "RandomAccessDatasetTrain" # It trains simple CNN model while load data.

### 5.4. Parallel listing of files or objects

Update the "workers" count in the following section , to start parallel listing.

```json
 "execution":{
    "workers": 10,  <<=== Update parallel workers no
    "steps":{
      "model": true,
      "training": true,
      "inference": false,
      "metrics": false
    }
  },
```

### 5.5. Parallel read of file or objects from S3

The PyTorch data_loader can be used to read files or objects in parallel through multiple workers. The workers number can
be specified in the following section.

```json
"PyTorch": {
                  "DataLoader": {
                    "shuffle": true,
                    "num_workers": 49,   << ==== Update this variable.
                    "persistent_workers": true,
                    "prefetch_factor": 2,
                    "pin_memory": false,
                    "drop_last": false
                  },
```

### 5.6. Update model

A default Convolution Neural Network (CNN) is provided with few layers. Based on that a custom class can be created.

```python
class ConvNet(NeuralNetwork):
    """
    Expected a image dimesion of 32x32 because of tensor size
    Train Data:torch.Size([64, 32, 32]),torch.Size([64])
    """
    def __init__(self,image_diments=()):
        super(ConvNet, self).__init__()
        self.input_image_height = int(image_diments[0])
        self.input_image_weidth = int(image_diments[1])
        self.network()


    def network(self):
        # <Input Channel>,<Output Channel>, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Feed 16 out channel from above convolution to linear layer with image dimensions 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)  # <input> ,<output>
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = x.unsqueeze(1)
        #print(x)
        # Max pooling over a (2,2) window ( Sub sampling ) of the output from first feature maps.
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

The same class name should be specified in the configuration file.

```json
"model":{
    "choice": "ConvNet",     

    "ConvNet": {
        "loss_function": "<Specify Loss Function ex.[CorssEntrophy]"
    },
    "ObjectDetector": {
        "loss_function": ["MSE", "CrossEntropy"]
    }
  }
```

### 5.7. Update training

A default training class is provided to train MapStyle/RandomAccess dataset. User supposed to write their own class and override train method.

```python
training.py
class RandomAccessDatasetTrain(DNNTrain):
    """
    Example of training for MapStyle dataset.
    """
    def __init__(self,**kwargs):
        self.config = kwargs["config"]
        self.train_dataloader = kwargs["dataloader"]
        super(RandomAccessDatasetTrain,self).__init__(config=self.config,
                                       model=kwargs["model"],
                                       device=kwargs["device"])

    def train(self):
        start_time = datetime.now()
        print(f"INFO: Training started! {start_time}")
        criterion = self.model.loss_function()
        optimizer = optimization.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(self.epochs):
            running_loss = 0.0
            # Following line returns image, label tensor.
            j = 0
            for batch_index, data in enumerate(self.train_dataloader, 0):
                images, labels = data
                images = images.float()  # Convert to float.
                # print(images[0])
                # Zero the parameter gradient
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(images)
                loss = criterion(outputs, labels)  # loss calculation based on CrossEntropy
                loss.backward()
                optimizer.step()

                # Add loss for 10 batches.
                running_loss += loss.item()
                if batch_index % self.max_batch_size == 0:
                    # print("Batch Index:{}, ImageTensor:{}, LabelTensor:{}".format(batch_index, len(images), len(labels)))
                    print(f'Epoch:{epoch + 1}, BatchIndex:{batch_index} loss: {running_loss / self.max_batch_size:.3f}')
                    running_loss = 0.0

        print("INFO: Training is done : {} seconds".format((datetime.now() - start_time).seconds))

```

Update custom training class name as below.

```json
"training":{
  "name": "RandomAccessDatasetTrain"  
},
```

### 5.8. Update Inference

Just like the `ObjectDetection` project, the inference part of the `ImageClassifier` has also been updated in a separate standalone script -- `image_classifier_predict.py`.
First to go for inference the Execution ***Config*** needs to be updated to `true`:

```json
 "execution":{
    "workers": 10,  
    "steps":{
      "model": true,
      "training": true,             
      "inference": false,              <<=== Update "true" here
      "metrics": false
    }
  },
```

-----------------------------------Run Instructions----------------------------------

The above script can be run with test images present on Filesystem or S3.

To run with test images from Filesystem:

```commandline
path]$ python3 image_classifier_predict.py --fs --input absolute/path/to/the/test/image.jpg
```

Or, for bulk image tests:

```commandline
path]$ python3 image_classifier_predict.py --fs --input 'absolute/path/to/the/text/file/containing/a/list/of/images.txt 
```

To run with test images from S3:

```commandline
path]$ python3 image_classifier_predict.py --s3 --input client_lib_name(dss_client / boto3):prefix/to/the/test/image.jpg\n '
```

Or, for bulk image tests:

```commandline
path]$ python3 image_classifier_predict.py --fs --input client_lib_name(dss_client / boto3):prefix/to/the/text/file/containing/a/list/of/images.txt\n'
```

**In case of bulk image tests with S3 inputs, please make sure that the "text" file' contains the list of the test images along with its full prefix.**

The output images would be saved under `image_classifier_outputs/predicted_images` directory within same project directory (`dss_ai_benchmark/`).

## 6. Workings of the Object Detection project

### 6.1. Add Data source (Dataset)

The object detection project should have data in the following structures.
The data source is supported with a set of categories (called labels) should be present in each data source. Here this project has been created to detect 5 categories of vehicles -- ["Car", "Motorcycle", "Airplane", "Bus", "Truck"]. It requires to provide source data of multiple images each of the above categories.
Each category directories should have their corresponding images. In the following example we have list down the directory structure.

```text
## File System Structure, specify with data_dir as list within Config file
# Source
Dataset/Car
Dataset/Motorcycle
Dataset/Airplane
Dataset/Bus
Dataset/Truck
```

Each of the above directories should have their corresponding images (1500 each while creation -- 7500 total images).

In addition, each of the above directories should also contain a subdirectory called ***Label***, which should have a list of `.txt` files with filenames same as each image filenames, present a directory above i.e. if there are 1500 images (.jpg) in `Dataset/Car` directory, there should be 1500 text files (.txt) in the `Dataset/Car/Label` directory, with the filenames being the same between the image & text files (like if image filename is `e652f30d67a222d6.jpg`, the corresponding text filename should be `e652f30d67a222d6.txt`). These text files contain the ***label*** and the ***bounding-box coordinates*** for each of the image (as per the filename). A sample text file content is shown below:

```commandline
OID]$ ls -lrt Dataset/Car/e652f30d67a222d6.jpg
-rwxrwxrwx 1 s.banerjee s.banerjee 269790 Jul 11  2018 Dataset/Car/e652f30d67a222d6.jpg
OID]$
OID]$ cat Dataset/Car/Label/e652f30d67a222d6.txt
Car 99.84 108.80025600000002 980.48 579.84
|_| |_____________________________________|----> bounding-box coordinates of where the object is appearing in the image (format: x_min y_min x_max y_max)
 |
 |----> category, the object in the image belonging to
```

Currently, this project only supports filesystem data. Update configuration accordingly, as shown in ***Section #3*** above.

### 6.2. CustomDataset class at the following file

```python
dataset/pytorch_dataset.py
class TorchObjectDetectionDataset(RandomAccessDataset):
    """
    User defined dataset class. The base class has all default functionalities
    Each functions in the override section can be overidden
    """

    def __init__(self, transforms=None, config={}, logger=None):
        super(TorchObjectDetectionDataset, self).__init__(transform=transforms,
                                                          config=config,
                                                          logger=logger)
        self.tensors = self.get_tensors()

    def __getitem__(self, index):

        ## Code here

        # return a tuple of the images, labels, and bounding box coordinates
        return image, label, bbox, img_read_time

    def __len__(self):
        return self.tensors[0].size(0)

    def get_tensors(self):
        ## Read all the different objects from each of the listed images....

        # grab the image, label, and its bounding box coordinates

        # Code here
        
        # convert NumPy arrays to PyTorch tensors and return the same
        images, labels, bboxes, read_times = torch.tensor(images), torch.tensor(labels), torch.tensor(bboxes), torch.tensor(read_times)

        return images, labels, bboxes, read_times
```

The **Config** file is as below:

```json
"TorchObjectDetectionDataset": {
                    "access": {"random": true, "sequential": false},
                    "type": "image",
                    "label": ["Car", "Motorcycle", "Airplane", "Bus", "Truck"],

                    "__comment__": "Specify ImageNet compatible image dimensions",
                    "image_dimension": [224, 224],

                    "__comment__": "Below line specifies the average image file size in 'KB'; has been included --",
                    "__comment__": "-- to avoid extra file reads to extract the dataset size.",
                    "avg_image_size": 315,

                    "__comment__": "specify ImageNet mean and standard deviation",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
               }
```

### 6.3. Update model

A pre-trained `ResNet50` model is used as the **base_model** which is then trained on the dataset through a Regressor model -- to predict bounding-box coordinates; and a Classifier model -- to classify the images into one of the 5 categories.

```python
class ObjectDetector(nn.Module):

    def __init__(self, image_diments=(), logger=None, num_classes=0):
        super(ObjectDetector, self).__init__()
        self.image_diments = image_diments
        self.num_classes = num_classes
        self.logger = logger

        # load the ResNet50 network
        self.baseModel = resnet50(pretrained=True)
        # freeze all ResNet50 layers, so they will *not* be updated during the training process
        for param in self.baseModel.parameters():
            param.requires_grad = False

        # build the regressor head for outputting the bounding box coordinates
        self.regressor = nn.Sequential(
            nn.Linear(self.baseModel.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )

        # build the classifier head to predict the class categories
        self.classifier = nn.Sequential(
            nn.Linear(self.baseModel.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, self.num_classes)
        )

        # set the classifier of our base model to produce outputs
        # from the last convolution block
        self.baseModel.fc = nn.Identity()

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network

        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)

        # return the outputs as a tuple
        return bboxes, classLogits
```

The same class name should be specified in the **Config** file.

```json
"model": {
    "choice": "ObjectDetector",     

    "ConvNet": {
        "loss_function": "<Specify Loss Function ex.[CorssEntrophy]"
    },
    "ObjectDetector": {
        "loss_function": ["MSE", "CrossEntropy"]
    }
  }
```

### 6.4. Update training

A default training class is provided to train MapStyle/RandomAccess dataset. We wrote our own class and overrode the default train method. Thismodule will also collate & produce the performance metrics -- to be stored in metrics.csv.

```python
training.py
class ObjectDetectionDatasetTrain(DNNTrain):

    def __init(self, **kwargs):
        super(ObjectDetectionDatasetTrain, self).__init__(**kwargs)

    def train(self):
        # define our loss functions
        classLossFunc = CrossEntropyLoss()
        bboxLossFunc = MSELoss()

        steps_per_epoch = len(self.train_dataloader.dataset) // self.config["framework"]["batch_size"]

        # initialize the optimizer, compile the model, and show the model summary
        opt = Adam(self.model.parameters(),
                   lr=self.config["training"][self.config["training"]["choice"]]["init_lr"])

        # initialize a dictionary to store training history
        history = {'total_train_loss': [], 'train_accuracy': []}

        self.logger.info("\nModel loaded and initial Training params setup completed....")
        time.sleep(0.1)

        start_time = time.monotonic()

        # Add metrics header
        self.metrics.append(
            ["num_epochs", "total_time (Sec)", "dataset_size (MBytes)", "read_bw (MB/Sec)", "dataset_read_time (Sec)"])
        tot_read_time, dataload_time, epoch_bw = [], [], []

        # loop over epochs
        self.logger.info("\nTraining the network....")
        for e in tqdm(range(self.epochs)):

            # set the model in training mode
            self.model.train()

            # initialize the total training and validation loss
            totalTrainLoss = 0

            # initialize the number of correct predictions in the training step
            trainCorrect = 0

            self.train_dataloader.dataset.dataset_size_in_bytes.value = 0

            epoch_start_time = time.monotonic()
            epoch_read_time = 0
            dload_elapsed = 0
            # loop over the training set
            for (images, labels, bboxes, img_read_times) in tqdm(self.train_dataloader):
                # send the input to the device
                (images, labels, bboxes) = (images.to(self.device),
                                            labels.to(self.device), bboxes.to(self.device))

                epoch_read_time += torch.mean(img_read_times).item()

                # perform a forward pass and calculate the training loss
                predictions = self.model(images)
                bboxLoss = bboxLossFunc(predictions[0], bboxes)
                classLoss = classLossFunc(predictions[1], labels)
                totalLoss = ((self.config["training"][self.config["training"]["choice"]]["bbox_loss"] * bboxLoss)
                             + (self.config["training"][self.config["training"]["choice"]]["label_loss"] * classLoss))

                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                opt.zero_grad()
                totalLoss.backward()
                opt.step()

                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += totalLoss
                trainCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()
                dload_elapsed += (time.monotonic() - epoch_start_time)
                epoch_start_time = time.monotonic()

            # calculate the average training loss
            avgTrainLoss = totalTrainLoss / steps_per_epoch

            # calculate the training accuracy
            trainCorrect = trainCorrect / len(self.train_dataloader.dataset)

            # update our training history
            history["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            history["train_accuracy"].append(trainCorrect)

            # print the model training and validation information
            self.logger.info("EPOCH: {}/{}".format(e + 1, self.epochs))
            self.logger.info("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avgTrainLoss, trainCorrect))

            tot_read_time.append(round(epoch_read_time, 4))
            dataload_time.append(round(dload_elapsed, 4))
            dataset_size_mb = round((self.train_dataloader.dataset.dataset_size_in_bytes.value / 1024), 2)
            epoch_bw.append(round((dataset_size_mb / epoch_read_time), 3))
        self.metrics.append([str(self.epochs).replace(',', ';'), str(dataload_time).replace(',', ';'),
                             str(dataset_size_mb).replace(',', ';'),
                             str(epoch_bw).replace(',', ';'), str(tot_read_time).replace(',', ';')])
        total_time = round(time.monotonic() - start_time, 4)
        bw = [round((self.train_dataloader.dataset.dataset_size_in_bytes.value / 1024) / t, 3) for t in tot_read_time]
        train_summary = "** Train Summary **\n"
        train_summary += "\t Epochs:{}, BatchSize:{}, MaxBatchSize:{}\n".format(self.epochs, self.batch_size,
                                                                                self.max_batch_size)
        train_summary += "\t Loading+Training Time:{:.2f} Sec, Dataset Size:{} KBytes, Read_BW:{} MiB/Sec, " \
                         "Total Read Time:{} Secs\n".format(total_time,
                                                            self.train_dataloader.dataset.dataset_size_in_bytes.value,
                                                            bw, tot_read_time)
        self.logger.info(train_summary)

        self.logger.info("\nTraining completed...")
```

The **Config** file should be updated as below:

```json
"training": {
      "choice": "ObjectDetectionDatasetTrain",

      "RandomAccessDatasetTrain": {
          "__comment__": "Whether we are going for a Random access or Sequential access",
          "access_method": "Random"
      },
      "ObjectDetectionDatasetTrain": {
          "__comment__": "Specify the Learning Rate, the initial Loss weights",
          "init_lr": 1e-4,
          "label_loss": 1.0,
          "bbox_loss": 1.0
      }
  },
```

### 6.5. (Optional) Update Inference

Once the training is complete, we can try the optional step of checking on how our Object Detection model is performing on unseen data.

A ***predict*** script `object_detector_predict.py` is added in the same project directory (`dss_ai_benchmark/`), which is a standalone script to provide inference on the trained model.

-----------------------------------Run Instructions----------------------------------

The above script can be run with test images present on Filesystem or S3.

To run with test images from Filesystem:

```commandline
path]$ python3 object_detector_predict.py --fs --input absolute/path/to/the/test/image.jpg
```

Or, for bulk image tests:

```commandline
path]$ python3 object_detector_predict.py --fs --input 'absolute/path/to/the/text/file/containing/a/list/of/images.txt 
```

To run with test images from S3:

```commandline
path]$ python3 object_detector_predict.py --s3 --input client_lib_name(dss_client / boto3):prefix/to/the/test/image.jpg\n '
```

Or, for bulk image tests:

```commandline
path]$ python3 object_detector_predict.py --fs --input client_lib_name(dss_client / boto3):prefix/to/the/text/file/containing/a/list/of/images.txt\n'
```

**In case of bulk image tests with S3 inputs, please make sure that the "text" file' contains the list of the test images along with its full prefix.**

The output images would be saved under `object_detector_outputs/predicted_images` directory within same project directory (`dss_ai_benchmark/`).
