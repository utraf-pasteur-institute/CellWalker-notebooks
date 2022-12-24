# Cellwalker-notebooks

This repository is a part of the CellWalker pipeline. It contains the IPython notebooks for automated segmentation and segmentation-visualization and export as 3D objects.
Contents:<br>
<ul>
<li> ./src/Segmentation_CNN_UNET.ipynb </li>
<li> Segmentation_visualizer.ipynb </li>
</ul>
<br>

# Automated segmentation
The notebook named 'Segmentation_CNN_UNET.ipynb' provides a protocol for automated segmentation of microscopy images using a UNET convolutional neural network (CNN) architecture.
It is recommended to run this notebook on cloud computing platforms such as Google Colab. This notebook has been tested on Google Colab.<br>

To open the notebook on Google Colab, upload it to your Google Drive and open it inside the browser.
Once open in Google Colab, the instructions in the notebook will guide you regarding functionality of each step. For each step, after setting required values, you can click on the 'Play' button to execute the step.

If you want to use the sample data provided in the repository then remember to upload the folders 'sample_data/C2-113_16-46_cropped1_for_UNET_training' and 'sample_data/C2-113_16-46_cropped3' to the Google Drive.<br>
In general any data that is on your Google Drive can be accessed from the notebooks opened in Google Colab.

For more information on how to use Google Colab, please click <a href="https://colab.research.google.com/" target="_blank">here</a>.

# Installation
The instructions assume Windows operating system. Most of these commands also work on linux systems.

This installation is required for the Segmentation_visualizer.ipynb
## Download and install Anaconda
Please see here to find more information.

## Create environment for CellWalker-notebooks
It is recommended to work inside an environment in order to avoid conflicts of package versions.<br>

Open Anaconda Prompt or Anaconda Powershell Prompt and run following commands

```bash
cd path_to_folder_where_you_downloaded_the_repository
```

```bash
conda env create -f cellwalker-notebooks-env
```

Optional: If you already have the Anaconda python environment with the same name then it is advised to remove the old environment and start fresh.
To list existing environments-
```bash
conda info --envs
```

To remove an environment
```bash
conda remove -n cellwalker-notebooks-env --all
```

## Activate the environment
```bash
conda activate cellwalker-notebooks-env
```
Now you will see the prompt changes from 'base' to 'cellwalker-notebooks-env'

## Add ipykernel to environment so that the environment can be used in Jupyter-notebook
```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=cellwalker-notebooks-env
```

## Deactivate the environment
```bash
conda deactivate
```

## Launch Jupyter-notebook
```bash
jupyter notebook
```
On linux, the command is ```jupyter-notebook```<Br>
Browse src folder inside the folder of the repository.<br>
Choose the notebook (.ipynb file) which you want to run.<br>
When the notebook opens, select the environment 'cellwalker-notebooks-env' from the menu option "Kernel>Change kernel".


# Troubleshooting
Sometimes Anaconda's default package versions do not allow ipython kernels to run. This will likely happen with Windows versions of Anaconda.
If you get an error related to the pyzmq package, then do the following.<br>

Uninstall pyzmq version that is installed in your environment by default. You can check the version by using pip list or conda list commands from Anaconda prompt on Windows.
Activate the required environment.
```bash
conda activate cellwalker-notebooks-env
```
If you want to see the installed packages and their versions-
```bash
conda list
```
Or you can also use 'pip list' to see the list of installed packages.

Uninstall pyzmq package.
```bash
conda uninstall pyzmq
```

Re-install pyzmq package so that it upgrades to the latest version.
```bash
conda install pyzmq
```

Install the ipykernel again because uninstalling pyzmq also uninstalls ipykernel related packages.
```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=cellwalker-notebooks-env
conda deactivate
```

Re-launch the Jupyter-notebook
```bash
jupyter notebook
```
On linux, the command is 'jupyter-notebook'.

