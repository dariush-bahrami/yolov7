# YOLOv7 Inference

*Note 1: This is a fork of YOLOv7 official [repo](https://github.com/WongKinYiu/yolov7)*

*Note 2: Check [usage.ipynb](https://github.com/dariush-bahrami/yolov7/blob/main/usage.ipynb)*

*Note 3: Check [yolov7.py](https://github.com/dariush-bahrami/yolov7/blob/main/yolov7.py)*

<hr /> 

Import required modules:

```python
import numpy as np
from PIL import Image

from yolov7 import YOLOv7, InferenceArgs
```

Create inference args object:

```python
args = InferenceArgs("weights/yolov7.pt")
```

Create yolov7 object:

```python
yolo = YOLOv7(args)
```

Load image:

```python
image = Image.open("inference/images/bus.jpg")
```

Pass image as numpy ndarray to the YOLOv7 object:

```python
prediction = yolo(np.asarray(image))
```

To visualize results use `.visualize()` method on prediction object:

```python
prediction.visualize()
```

