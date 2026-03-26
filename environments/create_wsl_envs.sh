#!/bin/bash

# how to (re-)create the local environments
ENV_NAME="watershed_workflow-`date +%F`"
python3 environments/create_envs.py --manager=micromamba --OS=Linux --env-type=USER --user-env-extras --with-user-env=watershed_workflow_user --with-tools-env=watershed_workflow_tools ${ENV_NAME}
micromamba run -n ${ENV_NAME} python3 -m ipykernel install \
        --name ${ENV_NAME} --display-name "Python3 ${ENV_NAME}"
micromamba env export -n ${ENV_NAME} --no-builds > environments/environment-wsl.yml


CI_ENV_NAME="watershed_workflow_CI-`date +%F`"
python3 environments/create_envs.py --manager=micromamba  --OS=Linux --env-type=CI ${CI_ENV_NAME}
micromamba run -n ${CI_ENV_NAME} python3 -m ipykernel install \
        --name ${CI_ENV_NAME} --display-name "Python3 ${CI_ENV_NAME}"
micromamba env export -n ${CI_ENV_NAME} --no-builds > environments/environment-CI-wsl.yml

DEV_ENV_NAME="watershed_workflow_DEV-`date +%F`"
python3 environments/create_envs.py --manager=micromamba --OS=Linux --env-type=DEV ${DEV_ENV_NAME}
micromamba run -n ${DEV_ENV_NAME} python3 -m ipykernel install \
        --name ${DEV_ENV_NAME} --display-name "Python3 ${DEV_ENV_NAME}"
micromamba env export -n ${DEV_ENV_NAME} --no-builds > environments/environment-DEV-wsl.yml

