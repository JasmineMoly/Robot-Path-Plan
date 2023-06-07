# Robot-Path-Plan
Path Planning Algorithm for 6-DOF Manipulator Applied in Medical Scenes
# 项目名称：六自由度机械臂路径规划算法研究


欢迎来到六自由度机械臂路径规划算法研究项目！这个项目旨在探索在医疗场景中应用的机械臂路径规划算法。我们利用RRT*（Rapidly-Exploring Random Trees）算法来规划机械臂的运动路径，以实现从起始点到目标点的安全导航。

## 背景
在医疗领域，机械臂被广泛应用于手术辅助、康复训练和药物配送等任务。然而，机械臂的路径规划是一个复杂的问题，需要考虑到患者和周围环境的安全性。因此，本项目旨在提供一种高效且可靠的路径规划算法，以确保机械臂的运动安全和精确性。

## 算法原理
我们使用Python编程语言实现了RRT*算法，并结合了3D可视化工具matplotlib来展示规划出的路径。以下是算法的主要步骤：

1. 初始化起始点和目标点。
2. 定义障碍物列表，其中每个障碍物由中心点和半径表示。
3. 生成随机点，用于扩展树结构。
4. 找到离随机点最近的节点，并以固定步长朝着随机点延伸，生成新的节点。
5. 检查新的节点与最近节点之间是否存在碰撞，以确保路径的安全性。
6. 如果新的节点通过了碰撞检查，将其添加到节点列表中，并在周围节点中重新连接路径以优化路径成本。
7. 重复上述步骤，直到达到最大迭代次数或找到可行路径。
8. 根据找到的最终节点和目标点，回溯生成完整的路径。

## 使用方法
你可以按照以下步骤使用该项目：

1. 安装Python和所需的依赖库（如numpy和matplotlib）。
2. 在`if __name__ == '__main__':`的代码块中，定义起始点、目标点和障碍物列表。
3. 根据需要调整RRT*算法的参数，如搜索半径、最大迭代次数和步长。
4. 运行代码，并观察控制台输出。如果找到有效路径，将显示“Found!”的消息。
5. 如果找到有效路径，将弹出一个包含起始点、目标点和路径的3D可视化图形窗口。

## 示例结果
在我们的示例中，我们

使用了一个起始点（1, 1, 1）、一个目标点（7, 10, 6）和四个障碍物。经过路径规划算法的运算，我们找到了一条从起始点到目标点的有效路径。在3D可视化图形窗口中，你可以清晰地看到起始点、目标点、障碍物和机械臂沿路径移动的过程。


## 注意事项
- 确保提供的起始点和目标点在可行的工作空间范围内，并避免与障碍物发生重叠。
- 如有需要，你可以根据自己的需求调整算法的参数，以获得更好的路径规划效果。

希望这个项目能够为你在医疗场景中应用机械臂路径规划算法提供一些参考和帮助。如果你对该项目有任何疑问或建议，请随时联系我们。感谢你的关注与支持！

*注意：以上示例代码仅为理解项目用途，实际应用中可能需要根据具体情况进行适当的修改和扩展。*
