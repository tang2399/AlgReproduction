# 已实现冒泡排序、选择排序、插入排序

# 冒泡排序
# 从最左边开始，依次比较相邻两个元素的大小，若左边的数大于右边的数就进行交换
def BubbleSort(L):
    List = L.copy()
    time = 0  # 比较次数
    changeTime = 0

    for i in range(1, len(List)):
        for j in range(0, len(List) - i):
            time += 1
            k = j + 1
            if List[j] > List[k]:
                changeTime += 1
                number = List[j]
                List[j] = List[k]
                List[k] = number

    print('\nBubbleSort using times:', time)
    print('BubbleSort change times:', changeTime)
    print('BubbleSort result', List)


# 选择排序
# 每次遍历时，将各个元素比较，将最大值或最小值的索引存放在一个变量中，全部比较完了以后，再将该索引上的元素进行交换
# 此处为依次找最小值
def SelectionSort(L):
    List = L.copy()
    time = 0
    changeTime = 0

    for i in range(0, len(List)):
        number = List[i]
        j = i + 1
        flag = i
        while j < len(List):
            time += 1
            if number > List[j]:
                number = List[j]
                flag = j
            j += 1
        if number != List[i]:
            changeTime += 1
            List[flag] = List[i]
            List[i] = number

    print('\nSelectionSort using times:', time)
    print('SelectionSort change times:', changeTime)
    print('SelectionSort result', List)


# 插入排序
# 在无序序列中任取一个数，插入有序序列
def InsertionSorting(L):
    List = L.copy()
    changeTime = 0

    flag = 1  # 有序序列中有flag+1个数
    while flag < len(List):
        if List[flag - 1] > List[flag]:  # 有序序列中最后一个数不有序
            for i in range(0, flag):
                if List[flag] < List[i]:  # 找到有序序列中最小的大于无序数的数
                    changeTime += 1
                    number = List[flag]
                    for j in range(flag, i - 1, -1):  # 有序序列往后腾
                        changeTime += 1
                        List[j] = List[j - 1]
                    List[i] = number
        flag += 1

    print('\nInsertionSorting change times:', changeTime)
    print('InsertionSorting result', List)


if __name__ == "__main__":
    l = [9, -5, 0, 3, 2]
    print('List:', l)
    BubbleSort(l)
    SelectionSort(l)
    InsertionSorting(l)
