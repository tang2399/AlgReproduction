def FIFO(pages):
    if pages is None:
        exit("wrong, pages are null.")

    frame = [-1] * 3  # 页帧
    flag = 0  # 将要被置换的帧下标
    error = 0  # 缺页错误数量

    for page in pages:
        # 用以判断页帧内是否已经存在将引用的页面
        nextPage = False

        # 判断是否已有该页
        for f in frame:
            if f == page:
                nextPage = True

        # 没有，引用并产生缺页错误
        if not nextPage:
            frame[flag] = page
            flag = [flag + 1, 0][flag == 2]  # 页帧下标不为2就+1，否则置0
            error += 1

    print('FIFO: number of errors:', error)


def LRU(pages):
    if pages is None:
        exit("wrong, pages are null.")

    frame = [-1] * 3  # 页帧
    frequency = [0] * 3  # 内存中的每页上次出现的时间，用以判断舍弃哪一页
    error = 0  # 缺页错误数量
    log = []  # 日志,每次引用就会把这次引用的页号加入到末尾，只会保存页面最新的引用记录

    for page in pages:
        # 删除该页上一次的引用记录
        for i in log:
            if i == page:
                log.remove(page)
        # 将该次引用计入日志
        log.append(page)

        nextPage = False
        # 判断内存内是否已经有该页，
        for f in range(len(frame)):
            # 读取内存中的页上一次使用的时间
            frequency[f] = frame.index(frame[f])

            if frame[f] == page:
                nextPage = True

        if not nextPage:
            # 找到最长时间没有使用的页，将其置换
            flag = max(frequency)
            frame[flag] = page
            error += 1

    print('LRU: number of errors:', error)


def OPT(pages):
    if pages is None:
        exit("wrong, pages are null.")

    # 将来要引用的页面列表，也就是等待队列
    waiting = []
    for page in pages:
        waiting.append(page)

    frame = [-1] * 3  # 页帧
    error = 0  # 缺页错误

    for page in pages:
        # 保存内存中每页下一次将要出现的位置，将其初始值设置的足够大
        frequency = [len(pages) + 1] * 3
        waiting.remove(page)
        nextPage = False

        for f in range(len(frame)):
            # 求内存中的页还有多久再一次引用
            for p in range(len(waiting)):
                if frame[f] == waiting[p]:
                    frequency[f] = p
                    break

            if frame[f] == page:
                nextPage = True

        if not nextPage:
            # 找到内存中离下次引用最久的页，并将其置换
            flag = frequency.index(max(frequency))
            frame[flag] = page
            error += 1
    print('OPT: number of errors:', error)


if __name__ == "__main__":
    # 页面引用串
    pageString = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3,
                  0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
    # pageString = [random.randint(0, 9) for i in range(10)]
    print('pages:', pageString)

    FIFO(pageString)
    LRU(pageString)
    OPT(pageString)
