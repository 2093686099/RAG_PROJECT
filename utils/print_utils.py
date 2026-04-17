
def _print_event(event: dict, _printed: set, max_length=1500):
    """
    打印事件信息，特别是对话状态和消息内容。如果消息内容过长，会进行截断处理以保证输出的可读性

    Args:
        event: dict, 事件信息，包含对话状态和消息内容
        _printed: set, 已打印的事件ID集合，用于避免重复打印
        max_length: int, 消息内容的最大长度，超过该长度会进行截断处理
    """
    current_state = event.get("dialog_state")
    if current_state:
        print("当前处于：", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]  # 如果消息是列表，则取最后一个
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + "...（已截断）"  # 超过最大长度，截断并添加提示
            print(msg_repr)  # 打印消息内容
            _printed.add(message.id)  # 记录已打印的事件ID
