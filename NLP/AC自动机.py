from collections import deque


class ACNode:
    """AC自动机节点"""

    def __init__(self):
        self.children = {}  # 子节点字典（字符到节点）
        self.fail = None  # 失败指针
        self.output = []  # 存储以该节点结尾的关键词
        self.length = 0  # 关键词长度（用于定位匹配位置）


class ACAutomaton:
    def __init__(self):
        self.root = ACNode()  # 根节点

    def add_patterns(self, patterns):
        """批量添加模式串"""
        for pattern in patterns:
            self._insert(pattern)

    def _insert(self, pattern):
        """插入单个模式串到Trie树"""
        node = self.root
        for char in pattern:
            if char not in node.children:
                node.children[char] = ACNode()
            node = node.children[char]
        node.output.append(pattern)  # 存储完整关键词
        node.length = len(pattern)  # 记录长度

    def build_failure_links(self):
        """构建失败指针（BFS遍历）"""
        queue = deque()

        # 第一层节点失败指针指向根节点
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)

        # 构建剩余节点的失败指针
        while queue:
            current_node = queue.popleft()

            for char, child in current_node.children.items():
                # 从父节点的失败指针开始查找
                fail_node = current_node.fail

                # 循环直到找到匹配或到达根节点
                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail

                child.fail = fail_node.children[char] if fail_node else self.root

                # 合并输出链（如"she"的失败指针指向"he"时需要包含两者）
                child.output += child.fail.output
                queue.append(child)

    def search(self, text):
        """执行多模式匹配"""
        results = []
        current_node = self.root

        for idx, char in enumerate(text):
            # 沿失败指针链查找匹配
            while current_node != self.root and char not in current_node.children:
                current_node = current_node.fail

            # 移动到子节点或根节点
            current_node = current_node.children.get(char, self.root)

            # 收集所有匹配到的关键词
            for pattern in current_node.output:
                start_pos = idx - len(pattern) + 1
                results.append((pattern, start_pos, idx))

        return results


if __name__ == "__main__":
    ac = ACAutomaton()

    # 添加需要匹配的关键词
    patterns = ["he", "she", "his", "hers", "hi"]
    ac.add_patterns(patterns)

    # 构建失败指针
    ac.build_failure_links()

    # 测试文本
    text = "ushershihe"

    # 执行匹配
    matches = ac.search(text)

    # 输出结果
    print("匹配到的关键词及位置（起始索引从0开始）：")
    for pattern, start, end in matches:
        print(f"'{pattern}' 出现在位置 {start}-{end}: {text[start:end + 1]}")