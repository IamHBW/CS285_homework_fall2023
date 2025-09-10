#! /bin/bash

# 将包含训练运行结果的 `data` 目录从远程实例同步到您的本地计算机。

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
echo "正在传输文件..."

rsync -av --progress -e "ssh -o StrictHostKeyChecking=no" ${ADMIN_USERNAME}@${PUBLIC_IP}:~/data ./data

echo "-------------------------------------"
echo "正在关闭..."
az vm deallocate --resource-group $RESOURCE_GROUP --name $INSTANCE_NAME
