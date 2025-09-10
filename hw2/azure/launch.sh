#! /bin/bash

# 在远程实例上启动一个命令。确保您的代码在此之前已同步到远程实例。

export RESOURCE_GROUP="cs285-rg"
export INSTANCE_NAME="cs285-vm"
export ADMIN_USERNAME=$(whoami)

echo "正在启动实例..."
az vm start --resource-group $RESOURCE_GROUP --name $INSTANCE_NAME

echo "-------------------------------------"
echo "正在等待实例启动..."

az vm wait --resource-group $RESOURCE_GROUP --name $INSTANCE_NAME --custom "instanceView.statuses[?code=='PowerState/running']"

PUBLIC_IP=$(az vm show -d -g $RESOURCE_GROUP -n $INSTANCE_NAME --query publicIps -o tsv)

while true; do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=1 ${ADMIN_USERNAME}@${PUBLIC_IP} "nvidia-smi" 2>/dev/null
  if [ $? -eq 0 ]; then
    break
  else
    sleep 1
  fi
done

echo "-------------------------------------"
echo "正在将文件传输到实例..."

rsync -av --progress -e "ssh -o StrictHostKeyChecking=no" . ${ADMIN_USERNAME}@${PUBLIC_IP}:~/

echo "-------------------------------------"
echo "正在运行命令..."

CMD="
  tmux new -d '
    export PATH=\"\$HOME/miniconda/bin:\$PATH\";
    export MUJOCO_GL=egl;
    $*;
    sleep 5m;
    sudo shutdown now'
"

echo $CMD

ssh -o StrictHostKeyChecking=no ${ADMIN_USERNAME}@${PUBLIC_IP} "$CMD"
