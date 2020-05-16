Simu1.2------
1.最短跳路由：多条最短路径，设法尽量随机选择，给OCS一些机会。(getpath()=1,无从下手)X
2.dragonfly重新规范编号。（simu1也改了）
3.添加linkweight（光1电2），尽量选择光路。
（和1.1比较，发现并没有什么不同。。。。。）

Simu1.2Test1-----
1. test file io
1). NetworkMatrix.txt: network flag + network matrix.
2). Delay (new delay appends on a new line).
2. Delay.txt file collects every mean ETE delay between every 2 nodes, there are 16 racks * 10 nodes = 160 nodes in total, so 
  there is a number of 160^2 data during one simulation.
3. co-simulation with Tensorflow.
1. OCS to rack, add a checkbox(queue) to find if the channel is available. v

Simu1.2Test2-----

2. Traffic could be changed. Provide a TrafficMatrix.txt file. v
3. Change actions from 4 to many...(4 classical networks ---to--- every 2 nodes of mid-topology) x
4. collect throughput, packet loss...  (V or X)
