1、在主从2台机器上检查或配置conda虚拟环境，激活环境后pip install -r requirements.txt进行安装，两者环境相同

2、检查和设置局域网环境，保证主从机器能互相通信，可以ping通

3、检查主机server.py脚本开头的两个数据文件路径，确保路径正确

4、修改client.py中SERVER_IP，保证和主机的实际局域网IP一致

5、在主从机上依次启动server.py和client.py，可以看到日志坐标和可视化位置

