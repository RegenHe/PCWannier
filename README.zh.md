# PCWannier

[English](README.md)

PCWannier 是一个根据数值本征模数据构造光子晶体 Wannier 紧束缚模型的 Python 程序。目前主要用于二维 Bloch 数据，可生成局域 Wannier 函数、hopping 矩阵、插值能带以及可选的拓扑计算结果。

## 安装

PCWannier 需要 Python 3.10 或更高版本。在项目目录中执行：

```bash
pip install -e .
```

如需安装可选的加速依赖，执行：

```bash
pip install -e ".[numba,performance]"
```

## 使用方法

首先准备一个 `incar` 输入文件，在其中设置晶格、k 点网格、能带窗口、投影函数，以及网格、场、材料和本征值文件的路径。数值数据需要由用户自行提供。

运行计算：

```bash
pcwannier -i path/to/incar --out path/to/output
```

也可以使用等价的模块运行方式：

```bash
python -m pcwannier -i path/to/incar --out path/to/output
```

常用选项：

```text
-t N                  使用 N 个工作线程
--backend auto         自动选择计算后端
--cache                复用已缓存的计算矩阵
-b                     绘制投影函数后退出
--interp points.txt    在指定的点网格上插值结果
```

运行 `pcwannier --help` 可查看全部命令行选项。常见输出包括 Wannier 函数、hopping 矩阵、能带数据和图片、拓扑图片以及运行日志。
