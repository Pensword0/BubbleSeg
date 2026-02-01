import json
import logging

# 这行使用 logging 模块（一个标准的Python日志记录工具）设置基本的日志配置。
# level=logging.INFO 表示日志记录器将捕获信息级别（INFO）及以上级别的日志消息。format='' 意味着日志消息将不使用特定的格式化。
logging.basicConfig(level=logging.INFO, format='')

class Logger:
    def __init__(self):
        # 类的初始化方法（__init__）创建了一个名为 entries 的字典，用于存储日志条目。每个条目将以一个唯一的键存储在这个字典中。
        self.entries = {}

    def add_entry(self, entry):
        # add_entry 方法允许向日志中添加一个新条目。新条目将添加到 entries 字典中，使用的键是当前字典长度加1（这样做为每个条目提供了一个唯一的序号）。
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        # 当打印一个 Logger 实例或将其转换为字符串时，__str__ 方法定义了其表现形式。这里它使用 json.dumps 方法将 entries 字典转换为格式化的JSON字符串。
        # sort_keys=True 表示字典将按键排序，indent=4 提供了一个漂亮的、易于阅读的格式。
        return json.dumps(self.entries, sort_keys=True, indent=4)
