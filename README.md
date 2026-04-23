# lmp_anly

一个基于LAMMPS的输出log、dipole.txt文件与reax_tool输出文件绘制figure的CLI应用。

## 功能

* [x] 基于LAMMPS log绘制figure
* [x] 基于LAMMPS log计算介电常数
* [x] 基于LAMMPS输出的dipole.txt计算介电常数
* [x] 统计 `reax_tool` 的输出文件并只绘制显著产物的数据
* [ ] 更多LAMMPS功能请在issue中提出

## 用法

lmp_anly现在有三个子命令，分别用于控制log文件中所有图像的绘制、介电常数的计算以及绘制reax_tool工具输出的物种与键结关系图像。

```bash
lmp_anly log log_file -f svg -e
```

`log` 命令现在有两个 `flag` 可以启用： `-f` 用于控制输出图像格式（png / svg），  `-e` 用于控制是否通过 log 文件计算介电常数。图像会输出到与 log 文件同目录的 figure 目录下（目录会自动创建），计算得到的介电常数则会打印在屏幕上以及写在与 log 文件相同目录下的 epsilon.txt 中。

```bash
lmp_anly epsilon dipole.txt
```

`epsilon` 功能虽然会用到 log 文件，但不需要手动输入 log 文件路径，它会在 dipole.txt 文件下寻找 log 文件，请在 LAMMPS `input` 文件中注意输出结构。同样，`epsilon` 的计算结果会在屏幕上打印显示，同时也会写入 `epsilon.txt` 文件中。

```bash
lmp_anly species output_dir --threshold n --timestep n --figformat svg
```

对于 `species` 命令，`--timestep / -t` 选项是必须的，在其中写入你 dump 文件的步长（**不是模拟步长**)。`--thershold / -th` 用于控制绘图的截断值，只有产量高于 n 的物种才会被绘制。

## 配置

lmp_anly采用`TOML`配置文件来管理需要从log中输出的折线图，配置文件会在第一次运行程序时在默认配置目录自动生成，在Linux下为`~/.config/lmp-anly/`，Windows下为`~/AppData/Local/lmp-anly/`，MacOS下为`"~/Library/Application Support/lmp-anly/"`。
配置文件包含`mpl_style`以及`line_element`两个大类，`mpl_style`包含`matplotlib.rcParam`的设置用于设置绘图样式，`line_element`则包含绘制折线图需要的所有元素。`mpl_style`的书写格式与`Python`中的命令如出一辙，`line_element`则是由`log`中该列的列名、纵轴标签、图例和图文件名组成。其中图文件名`fig_name`允许在多个设置中重复，`fig_name`重复的线会被绘制在一张图上，此时只有第一次出现的`ylabel`是有效的。

```toml
[mpl_style]
"mathtext.fontset" = "stix"
"figure.constrained_layout.use" = true
"figure.dpi" = 300
"lines.linewidth" = 0.35
"font.family" = ["Nimbus Roman", "SimSun"]
"figure.figsize" = [3.54, 2.36]
[line_element.density]
column_name = "Density"
ylabel = "Density ($\\mathrm{g/cm^3}$)"
label = "Density"
fig_name = "density"
```

由于默认的`mpl_style`中包含了宋体，因此请注意电脑中是否有安装该字体。

目前根据作者本人的需求已经包括了：几乎所有的能量、密度、压强、体积、温度、均方根偏差、电偶极矩(非默认列名安装作者习惯设置)，基本所有默认的`thermo`输出都已囊括在其内，用户自定义的变量需要自行添加。

## 鸣谢

[matplotlib](https://matplotlib.org)

[pandas](https://pandas.pydata.org/)

[NumPy](https://numpy.org)

[SciPy](https://scipy.org)

[Typer](https://typer.tiangolo.com/)

## 许可

AGPL-3.0
