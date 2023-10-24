#!/bin/bash --login
# install reqs
cd $REPO_PATH

USER_PIP=$(which pip)

if [ ! -z ${YUM_REQUIREMENTS+x} ]
then
    # apt install them
    xargs -d '\n' -- sudo yum install -y < $YUM_REQUIREMENTS
fi

if [ ! -z ${PYTHON_REQUIREMENTS+x} ]
then
    # pip install them
    $USER_PIP install -r $PYTHON_REQUIREMENTS
fi

if [ ! -z ${CONDA_REQUIREMENTS+x} ]
then 
	# conda install them
	conda env update -f $CONDA_REQUIREMENTS
fi

cd /home/datascience/
