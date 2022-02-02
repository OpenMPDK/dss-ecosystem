# dss_dnn_benchmark
A custom benchmark tool to evaluate read access of DSS Object storage through deep neural networks such as PyTorch , TensorFlow.
The tool gives the flexibility to the user to add their custom dataset, training and neural network based model.These classes
can be written into the respective files and can be configured through configuration file. This tool is specifically meant
for classification problem where user suppose to specify data for couple of categories, train model and optional inference.

Pending works
- Metrics collection system development
- Inference ( Test data set and sample inference for end to end flow )
## Setup and Installation
Install the required libraries by running the following command in the node you are working on.
```
python3 -m pip install -r requirements.txt
```

## Configuration
The custom usage requires user to update dataset, model and training. Update the following sections.
- Set the deep neural network. ( Supported PyTorch and TensorFlow)
- Set data source
- Add custom dataset class name
- Add neural network name
- Add training class name 
## How to add custom dataset
### Data source 
As mentioned above, the tool is suitable for classification problem which should have data in the following structures.
The multiple data source is supported with same set of categories present into each data source. Let say we want to solve
a classification problem of flowers. It requires to provide source data of multiple type of flowers. These are called categories.
Each flower directories should have images. In the following example we have five categories of flower images.
```
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

## DSS Should hide bucket frrom user. So, bucket name is not relevant for that 
```
The tool support file system data as well as S3 data. Update configuration accordingly.
For AWS, we are suppose to add bucket name (bucket).
```
"storage": {
      "format": "fs",
      "name": "nfs1",
      "s3": {
        "bucket": "bucket",
        "prefix": ["flower_photos/", "flower_photos2/"],
        "client_lib": "boto3",
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
         "data_dir": ["/home/somnath.s/.keras/datasets/flower_photos","/home/somnath.s/.keras/datasets/flower_photos2"]
      }
```
### Create a CustomData class at  the following file
```
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
Update the configuration file as below.
If using MapStyle/RandomAccess , then set random to true.
Use the image dimension to 32x32 , or update into torch size.
Update categories to read from.

```
"dataset" : {
             "name": "MyDataset",
             "access": {"random": true, "sequential":  false},
             "type":  "image",
             "label": ["roses", "sunflowers", "tulips", "daisy", "dandelion"],
             "image_dimension": [32,32]
            },

## Add / update training class
```
## Parallel listing of files or objects
Update the "workers" count in the following section , to start parallel listing.
```
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
## Parallel read of file or objects from S3
The PyTorch data_loader can be used to read files or objects in parallel through multiple workers. The workers number can
be specified in the following section. 
```
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
## Update model
A default Convolution Neural Network (CNN) is provided with few layers. Based on that a custom class can be created.

```
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
```
"model":{
  "name": "ConvNet",  <<== Add your neural network class name.
  "loss_function": "<Specify Loss Function ex.[CorssEntrophy]"
},

```

## Update training
A default training class is provided to train MapStyle/RandomAccess dataset. User supposed to write their own class and override train method.

```
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
```
"training":{
  "name": "RandomAccessDatasetTrain"  
},
```

## Execution
Once all the configuration is done, run the tool as below.
```
python3 benchmark.py

```


