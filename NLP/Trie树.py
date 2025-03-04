# 节点的实现
class Node(object):
    def __init__(self, value: object) -> None:
        """
        初始化节点
        value: 表示该节点存储的值
        """
        self._children = {}  # 键为字符，值为 Node 对象
        self._value = value  # 标记词语存在

    def _add_child(self, char, value, overwrite=False):
        """
        添加子节点方法
        char: 字符类型，子节点对应的字符
        value: 子节点初始化的值（仅在新建时使用）
        overwrite: 是否覆盖现有子节点的值（默认不覆盖）
        """
        child = self._children.get(char)  # 获取当前字符对应的子节点

        if child is None:
            # 子节点不存在，创建新节点并添加到子节点字典
            child = Node(value)
            self._children[char] = child
        elif overwrite:
            child._value = value

        return child  # 返回子节点，便于链式调用构建字典树


# 字典树实现
class Trie(Node):
    def __init__(self) -> None:
        super().__init__(None)  # 根节点的值为None

    def __contains__(self, key):
        """
        检查最终节点是否为 None，以此检查 key 是否存在于字典树中
        key: 类似字典的键
        """
        return self[key] is not None

    def __getitem__(self, key):
        """获取 key 对应的值"""
        state = self
        for char in key:
            state = state._children.get(char)
            if state is None:
                return None
        return state._value

    def __setitem__(self, key, value):
        state = self
        for i, char in enumerate(key):
            if i < len(key) - 1:
                state = state._add_child(char, None, False)
            else:
                state = state._add_child(char, value, True)

    def items(self, prefix=None):
        result = []
        if prefix is None or prefix == "":
            # 收集所有键值对
            self._collect_all(self, "", result)
        else:
            # 定位前缀节点
            node = self._find_prefix_node(prefix)
            if node is not None:
                self._collect_all(node, prefix, result)
        return result

    def _find_prefix_node(self, prefix):
        """寻找前缀节点，进行前缀定位"""
        node = self
        for char in prefix:
            node = node._children.get(char)
            if node is None:
                return None
        return node

    def _collect_all(self, key, prefix, result):
        """收集当前节点的所有子节点值"""
        if key._value is not None:
            result.append((prefix, key._value))
        for char, child in key._children.items():
            self._collect_all(child, prefix + char, result)


if __name__ == '__main__':
    trie = Trie()
    # 增
    trie['自'] = 'I'
    trie['自然'] = 'nature'
    trie['自然语言'] = 'language'
    trie['自语'] = 'talk to oneself'
    trie['入门'] = 'introduction'
    trie['自然人'] = 'human'
    trie['入'] = 'in'
    # 改
    trie['自然语言'] = 'human language'
    print(trie.items())
    print(trie.items('自然'))
    # assert '自然' in trie  # 断言，为假输出一条错误信息
    # # 删
    # trie['自然'] = None
    # assert '自然' not in trie
