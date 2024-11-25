# SSTAR

![alt text](https://github.com/SafwenNaimi/SSTAR/blob/main/architecture.png)

This repository contains the official implementation for the paper "SSTAR: Skeleton-based Spatio-temporal action recognition for intelligent video surveillance and suicide prevention in metro stations"

# Installation

* tensorflow 2.6.0
* Python 3.8.5
* numpy 1.23.5

Clone this repository.

    git clone git@github.com:GIT-USERNAME/SSTAR.git
    cd SSTAR

Clone the repository and install the required pip packages (We recommend a virtual environment):

    pip install -r requirements.txt

# Notice Regarding ARMM dataset Usage:

Dear Researchers and Collaborators,

, you can directly download them and use them for training & testing.

As part of our commitment to advancing research, we have provided a benchmark skeleton dataset. While we are excited to share this resource with the community, we want to highlight some important considerations:

Copyright Compliance: This dataset is shared for research purposes only. Please ensure that your use of the dataset complies with all applicable copyright laws and regulations. Redistribution of the dataset without proper authorization is not permitted.

Privacy Considerations: We have taken steps to ensure the privacy and confidentiality of any individuals or entities represented in the dataset. We only provide the skeleton ground truth annotations of our ARMM dataset.

Intended Use: This dataset is intended solely for academic and research purposes. We urge users to apply this data responsibly and ethically, keeping in mind its intended application in the field.

User Agreement: By accessing and using this dataset, users agree to adhere to these terms and conditions. Misuse of the data or violation of these guidelines may result in restricted access to future resources.

We appreciate your cooperation in using this dataset responsibly. If you have any concerns or questions regarding the dataset, please feel free to contact us through GitHub.

Thank you for supporting responsible research practices.



# Training:
To run a baseline SSTAR experiment:

    python main.py -b 
