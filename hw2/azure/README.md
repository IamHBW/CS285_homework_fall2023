
## 使用 Azure

这里我们提供4个脚本来帮助您在Azure上运行实验。

### 1. 安装 Azure CLI

请按照[此处的说明](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)安装Azure CLI。然后，运行 `az login` 登录。

### 2. 创建虚拟机

运行 `azure/create_instance.sh` 来创建一个虚拟机实例。**请确保从 `hw2` 目录运行所有脚本。** 这将创建一个名为 `cs285-vm` 的实例，该实例配备一块 NVIDIA T4 GPU、4个CPU，并预装了带有PyTorch的Data Science Virtual Machine映像。

`create_instance.sh` 脚本还将等待NVIDIA驱动程序准备就绪，将您当前的目录（应为`hw2`）传输到虚拟机，安装所有依赖项，然后关闭虚拟机。您可以通过运行 `az vm list` 来查看新创建的虚拟机。

### 3. 运行作业

运行作业的主要入口点是 `launch.sh`。您可以通过在命令前添加脚本来使用它：例如 `azure/launch.sh bash sweep_lambda.sh`。这将执行以下操作：

1. 启动虚拟机
2. 将您当前的目录（应为`hw2`）同步到虚拟机
3. 启动一个新的tmux会话，以防止在您暂停本地计算机或与虚拟机的连接断开时作业被终止

在tmux会d话内部，会发生以下情况：

1. 您的命令将运行直到完成
2. shell将休眠5分钟
3. 虚拟机会被关闭

shell休眠5分钟的原因是，如果您的作业立即崩溃，您有时间在虚拟机关闭之前查看错误消息。

### 4. 检查您的作业

运行 `azure/launch.sh` 后，您可以通过运行 `ssh <username>@<public_ip>` 连接到正在运行的虚拟机。连接后，运行 `tmux ls` 查看正在运行的tmux会话。应该只有一个，但请记住，多次运行 `launch.sh` 可能会意外启动多个会话。要连接到tmux会话，请运行 `tmux a -t <session_id>`（如果只有一个会话，可以省略 `-t <session_id>`）。要从会话中分离，请按 `Ctrl-b d`。

进入tmux会话后，您应该会看到您的命令正在运行。如果您的命令立即崩溃，您可以看到错误消息，并且在虚拟机关闭前有5分钟的时间。要取消此关闭，请执行 `Ctrl-b :kill-pane`。**但是，您必须自己关闭虚拟机，或者再次运行 `launch.sh` 以启动另一个将在完成后自动关闭的作业。不要让您的实例一直运行！**

### 5. 启动TensorBoard

您可以通过运行 `azure/launch_tensorboard.sh` 在虚拟机上运行TensorBoard。然后，您应该能够通过在本地计算机上访问 `http://localhost:6006` 来查看TensorBoard。这仅在您保持脚本运行时有效。

### 6. 将数据同步到本地计算机

作业完成后，您可以通过运行 `azure/sync_data.sh` 将实验输出同步回本地计算机。这将启动实例，将远程 `data/` 目录下载到本地 `data/` 目录，然后再次关闭它。


## 重要说明

- **首先，也是最重要的，不要让您的实例一直运行！它运行的每个小时都会向您收费，即使您没有使用它。**
- 请记住从 `hw2` 目录运行所有脚本。
- 您可能只需要实例来进行大规模的超参数扫描。运行单个Python命令会相当浪费，而且在您的本地计算机上可能更快（即使您没有GPU！）。相反，请使用实例一次运行多个您在shell脚本中定义的作业：例如，`azure/launch.sh bash sweep_lambda.sh`。要一次运行多个作业，您可以使用子shell（即 `&`）或[GNU parallel](https://www.gnu.org/software/parallel/)（已在虚拟机上预装）。如果使用子shell，请确保在shell脚本的末尾放置 `wait`，否则虚拟机将关闭而不会等待您的作业完成。

