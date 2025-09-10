#! /bin/bash

# 在远程实例上启动tensorboard，并将端口转发到您的本地计算机。

export RESOURCE_GROUP="cs285-rg"
export INSTANCE_NAME="cs285-vm"
export ADMIN_USERNAME=$(whoami)

PUBLIC_IP=$(az vm show -d -g $RESOURCE_GROUP -n $INSTANCE_NAME --query publicIps -o tsv)

ssh -o StrictHostKeyChecking=no ${ADMIN_USERNAME}@${PUBLIC_IP} 'sudo pkill -f tensorboard'
ssh -o StrictHostKeyChecking=no -L 6006:localhost:6006 ${ADMIN_USERNAME}@${PUBLIC_IP} 'export PATH="$HOME/miniconda/bin:$PATH"; tensorboard --logdir data --port 6006'
