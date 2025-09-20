#! /bin/bash

# 创建一个新的GPU实例，将代码传输到该实例，并运行一些安装步骤。

export RESOURCE_GROUP="cs285-rg"
export LOCATION="koreacentral"
export INSTANCE_NAME="cs285-vm"
export VM_SIZE="Standard_NC4as_T4_v3"
# 使用Ubuntu 22.04 LTS with GPU支持
export IMAGE="Ubuntu2204"
export ADMIN_USERNAME=$(whoami)

echo "正在创建资源组..."
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "正在创建实例..."

az vm create \
  --resource-group $RESOURCE_GROUP \
  --name $INSTANCE_NAME \
  --size $VM_SIZE \
  --image $IMAGE \
  --admin-username $ADMIN_USERNAME \
  --generate-ssh-keys

echo "-------------------------------------"
echo "正在等待NVIDIA驱动程序安装..."

while true; do
  output=$(az vm run-command invoke -g $RESOURCE_GROUP -n $INSTANCE_NAME --command-id RunShellScript --scripts "nvidia-smi" --query "value[0].message" -o tsv 2>&1)

  if [[ $output == *"NVIDIA-SMI"* ]]; then
    echo "$output"
    break
  else
    echo "驱动程序尚未就绪，正在重试..."
    sleep 10
  fi
done

echo "-------------------------------------"
echo "正在将文件传输到实例..."

PUBLIC_IP=$(az vm show -d -g $RESOURCE_GROUP -n $INSTANCE_NAME --query publicIps -o tsv)

rsync -av --progress -e "ssh -o StrictHostKeyChecking=no" . ${ADMIN_USERNAME}@${PUBLIC_IP}:~/

echo "-------------------------------------"
echo "正在安装驱动..."

az vm run-command invoke -g $RESOURCE_GROUP -n $INSTANCE_NAME --command-id RunShellScript --scripts '
  # 安装NVIDIA驱动和CUDA
  sudo apt-get update
  sudo apt-get install -y ubuntu-drivers-common
  sudo ubuntu-drivers autoinstall
  
  # 安装依赖包
  sudo apt-get install -y swig python3-dev python3-pip parallel wget
  
  # 安装Miniconda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  echo "export PATH=\"$HOME/miniconda/bin:\$PATH\"" >> ~/.bashrc
  source ~/.bashrc
  
  # 设置环境变量
  echo "export MUJOCO_GL=egl" >> ~/.bashrc
  
  # 安装Python依赖
  $HOME/miniconda/bin/pip install -r requirements.txt
  $HOME/miniconda/bin/pip install -e .
'

echo "-------------------------------------"
echo "正在关闭..."
az vm deallocate --resource-group $RESOURCE_GROUP --name $INSTANCE_NAME
