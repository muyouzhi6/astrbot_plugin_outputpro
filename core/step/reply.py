from astrbot.core.message.components import (
    At,
    Face,
    Image,
    Plain,
    Reply,
)

from ..config import PluginConfig
from ..model import OutContext, StepName, StepResult
from .base import BaseStep


class ReplyStep(BaseStep):
    name = StepName.REPLY
    unsupported_platforms = {"dingtalk"}

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.cfg = config.reply

    async def handle(self, ctx: OutContext) -> StepResult:
        platform_name = ctx.event.get_platform_name()
        if platform_name in self.unsupported_platforms:
            return StepResult(msg=f"平台不支持智能引用，已跳过: {platform_name}")

        if self.cfg.threshold > 0 and all(
            isinstance(x, Plain | Image | Face | At) for x in ctx.chain
        ):
            msg_id = ctx.event.message_obj.message_id
            queue = ctx.group.msg_queue
            if msg_id in queue:
                pushed = len(queue) - queue.index(msg_id) - 1
                if pushed >= self.cfg.threshold:
                    ctx.chain.insert(0, Reply(id=msg_id))
                    if self.cfg.include_at:
                        ctx.chain.insert(1, At(qq=ctx.event.get_sender_id()))
                        # 在 At 后添加带零宽空格包裹的空格，确保与后续内容有间距
                        ctx.chain.insert(2, Plain(text="\u200b \u200b"))
                    queue.clear()
                    return StepResult(msg=f"已插入Reply组件, 引用消息{msg_id}")
        return StepResult()
