## Solis

Solis is Atlas' open source image segmentation and classification software. It is called Solis because it was built and trained on solar installations, but it can be used for any objects recognizable in satellite imagery. This package includes

### Setting up the project

It is recommended to install the project using a virtual python environment. Set it up and activate it by running these commands

`python -m venv venv`

`source ./venv/bin/activate`

To install the dependencies of the project, run

`pip install .`

Because Solis runs using cudnn, it is highly recommended to run the project using a nvidia GPU. If not, the classification and segmentation speeds will be very slow.

### Running the project

The magic of Solis is how simple it is to run, simply using your terminal. Done this way, Solis can be used on any machine, local or remote, in the cloud or in any environment of your choice. In Atlas, we often run Solis on a remote machine in a dedicated process. This lets us continously inspect the training results, without needing to have our own machine running a critical process at all times. You must make sure that you have used our s2dataset package to download and process Sentinel-2 data, and have included the ground truth files before running Solis. Make sure your virtual environment is installed and activated. Solis is meant to be run using the CLI. See the possible options by running:

`python ./src/solis --help`

Using the CLI, you can select many different models, prediction styles as well as training preferences. Define them in yaml or json using the Lightning protocol. The possible preferences are as follows

#### Datamodules

- S2Classification
- S2Segmentation

#### Neural network configurations

- ResNet34
- ResNet50
- ResNet101
- DeepLabV3_ResNet50
- DeepLabV3_ResNet101

#### Training preferences

- Fit
- Validate
- Test
- Predict
