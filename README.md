
**EQpolarity**
======

## Description

**EQpolarity** package is a deep-learning-based package for determining earthquake first-motion polarity

## Reference
Chen et al., 2023, Deep learning for P-wave first-motion polarity determination and its application in focal mechanism inversion.

	Chen Y, Saad OM, Savvaidis A, Zhang F, Chen Y, Huang D, Li H, Zanjani FA, 2024, Deep learning for P-wave first-motion polarity determination and its application in focal mechanism inversion. IEEE Transactions on Geoscience and Remote Sensing, 62, 5917411.

BibTeX:

	@article{eqpolarity,
	  title={Deep learning for P-wave first-motion polarity determination and its application in focal mechanism inversion},
	  author={Yangkang Chen and Omar M. Saad and Alexandros Savvaidis and Fangxue Zhang and Yunfeng Chen and Dino Huang and Huijian Li and Farzaneh Aziz Zanjani},
	  journal={IEEE Transactions on Geoscience and Remote Sensing},
	  year={2024},
	  volume={62},
	  number={1},
	  pages={5917411},
	  doi={10.1109/TGRS.2024.3407060}
	}
 
-----------
## Copyright
    Developers of the EQpolarity package, 2021-present
-----------

## License
    MIT License

-----------

## Install
First set up the environment and install the dependency packages

	conda create -n eqp python=3.11.7
	conda activate eqp
	conda install ipython notebook
	pip install matplotlib==3.8.0 tensorflow==2.16.2 scikit-learn==1.2.2 seaborn==0.12.2
	
Then install eqpolarity using the latest version

    git clone https://github.com/chenyk1990/eqpolarity
    cd eqpolarity
    pip install -v -e .
    
Or using Pypi

	pip install eqpolarity
	
Or using pip directly from Github

	pip install git+https://github.com/chenyk1990/eqpolarity
    
-----------
## Examples
# Texas Data Example
https://mega.nz/file/chxx1Z5Y#zXNRKT5aeNy7AGREKEUIq71TREK8hcUyXA1ZOkQ9DlM

-----------
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, or collaborations, please contact  
    Yangkang Chen
    chenyk2016@gmail.com

-----------
## NOTES:

-----------
## Gallery
The gallery figures of the eqpolarity package can be found at
    https://github.com/chenyk1990/gallery/tree/main/eqpolarity

These gallery figures are also presented below. 


