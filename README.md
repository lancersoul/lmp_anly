# lmp_anly

一个基于LAMMPS的输出log、dipole.txt文件与reax_tool输出文件绘制figure的CLI应用。

## 功能

* [x] 基于LAMMPS log绘制figure
* [x] 基于LAMMPS log计算介电常数
* [x] 基于LAMMPS输出的dipole.txt计算介电常数
* [x] 统计 `reax_tool` 的输出文件并只绘制显著产物的数据
* [ ] 更多LAMMPS功能请在issue中提出

## 用法

```bash
lmp_anly log log_file -f svg -e
```

lmp_anly现在有两个 `flag` 可以启用： `-f` 用于控制输出图像格式（png / svg），  `-e` 用于控制是否通过 log 文件计算介电常数。图像会输出到与 log 文件同目录的 figure 目录下（目录会自动创建），计算得到的介电常数则会打印在屏幕上以及写在与 log 文件相同目录下的 epsilon.txt 中。

```bash
lmp_anly epsilon dipole.txt
```

`epsilon` 功能虽然会用到 log 文件，但不需要手动输入 log 文件路径，它会在 dipole.txt 文件下寻找 log 文件，请在 LAMMPS `input` 文件中注意输出结构。同样，`epsilon` 的计算结果会在屏幕上打印显示，同时也会写入 epsilon.txt 文件中。

```bash
lmp_anly species output_dir --threshold n --timestep n --figformat svg
```

对于 `species` 命令，`--timestep / -t` 选项是必须的，在其中写入你 dump 文件的步长（**不是模拟步长**)。`--thershold / -th` 用于控制显示门槛，产量高于此值的物种才会被纳入统计并在图像中显示。

## 鸣谢

[matplotlib](https://matplotlib.org)

[pandas](https://pandas.pydata.org/)

[NumPy](https://numpy.org)

[SciPy](https://scipy.org)

[Typer](https://typer.tiangolo.com/)

## 许可

AGPL-3.0
