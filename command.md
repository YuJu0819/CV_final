> conda config --set channel_priority flexible
> conda install -c conda-forge opencv
> conda config --add channels conda-forge
> conda update conda
> conda install -c menpo opencv

> conda create --name new_cv_env python=3.9
> conda env remove --name env_name  