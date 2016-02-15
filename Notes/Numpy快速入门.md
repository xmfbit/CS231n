## Numpy 快速入门

###  基础知识

- Numpy 的主要数据结构是对齐的多维矩阵。它是由相同类型的元素组成的，索引的时候使用tuple，就像这样：

```
ele = a[0, 1]   # ele 是 array a 的 (0,1) 元素。注意，python和MATLAB不同，索引值是从0开始的
```

- Numpy 的矩阵类型是`ndarry`或是`array`。常用的成员属性有

    1. `ndim`: 返回矩阵的维数，如2维矩阵就是由若干行列组成的二维数组
    2. `shape`: 返回矩阵各个维度的大小，和MATLAB的`size`函数对应。不难看出，`ndim=len(shape)`
    3. `size`: 返回矩阵的元素个数，和MATLAB的`numel`函数对应。
    4. `dtype`: 返回矩阵元素的数据类型。比如`np.int32`, `np.int16`, `np.float64`等。
    5. `itemsize`: 返回矩阵单个元素所占的字节数，和C语言中的`sizeof`运算符是一样的。
    
### 创建矩阵

- 从python的list或者tuple中可以创建矩阵，使用`array`函数就可以了。完整声明如下：

```
numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)
```

    我们需要传给这个函数一个list类型的参数，还可以用dtype参数指定元素的类型。

- 特殊矩阵

    1. `zeros`: 创建全零矩阵，和MATLAB的`zeros`函数相同。
    2. `ones`: 创建全1矩阵，和MATLAB的`ones`函数相同。
    3. `empty`: 按照给定大小创建一个随机初始化的矩阵，给它分配空间
    4. `arange`: 和python的range类似，创建一个序列 (start, end, step)
    5. `linspace`: 和MALTAB的linspace相同，创建平均分布的点
    
    传递给这几个函数的参数可以是用tuple给定的矩阵大小，还可以使用dtype指定元素的类型。 

  