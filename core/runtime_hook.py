from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import wraps
from typing import Any

from astrbot.api import logger
from astrbot.core.message.components import BaseMessageComponent
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.platform import Platform

from .config import PluginConfig
from .sanitize import collect_visible_text, replace_in_chain, sanitize_chain


def _iter_subclasses(cls: type) -> list[type]:
    seen: set[type] = set()
    pending = [cls]
    ordered: list[type] = []

    while pending:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        pending.extend(current.__subclasses__())

    return ordered


def _normalize_message(message: Any) -> MessageChain | None:
    if message is None:
        return None
    if isinstance(message, MessageChain):
        return message
    if isinstance(message, str):
        return MessageChain().message(message)
    if isinstance(message, list) and all(
        isinstance(component, BaseMessageComponent) for component in message
    ):
        return MessageChain(chain=message)
    return None


class RuntimeOutputHook:
    def __init__(self, config: PluginConfig):
        self.config = config
        self._patched: list[tuple[type, str, Any]] = []
        self._installed = False

    def install(self) -> None:
        for cls in _iter_subclasses(AstrMessageEvent)[1:]:
            self._patch_event_send(cls)
            self._patch_event_streaming(cls)

        for cls in _iter_subclasses(Platform)[1:]:
            self._patch_platform_send_by_session(cls)

        self._installed = True

    def uninstall(self) -> None:
        for cls, attr, original in reversed(self._patched):
            setattr(cls, attr, original)
        self._patched.clear()
        self._installed = False

    def _patch(self, cls: type, attr: str, wrapper_factory) -> None:
        original = getattr(cls, attr, None)
        if original is None or getattr(original, "__outputpro_wrapped__", False):
            return

        wrapped = wrapper_factory(original)
        wrapped.__outputpro_wrapped__ = True
        self._patched.append((cls, attr, original))
        setattr(cls, attr, wrapped)

    def _patch_event_send(self, cls: type) -> None:
        def factory(original):
            hook = self

            @wraps(original)
            async def wrapped(event_self, message, *args, **kwargs):
                chain = _normalize_message(message)
                if chain is not None:
                    if hook._filter_chain(chain, source=f"{cls.__name__}.send"):
                        return None
                    message = chain
                return await original(event_self, message, *args, **kwargs)

            return wrapped

        self._patch(cls, "send", factory)

    def _patch_event_streaming(self, cls: type) -> None:
        def factory(original):
            hook = self

            @wraps(original)
            async def wrapped(event_self, generator, *args, **kwargs):
                if generator is None:
                    return await original(event_self, generator, *args, **kwargs)

                async def filtered_generator() -> AsyncGenerator[Any, None]:
                    async for item in generator:
                        chain = _normalize_message(item)
                        if chain is None:
                            yield item
                            continue

                        if hook._filter_chain(
                            chain,
                            source=f"{cls.__name__}.send_streaming",
                        ):
                            continue

                        if not chain.chain:
                            continue

                        yield chain

                return await original(event_self, filtered_generator(), *args, **kwargs)

            return wrapped

        self._patch(cls, "send_streaming", factory)

    def _patch_platform_send_by_session(self, cls: type) -> None:
        def factory(original):
            hook = self

            @wraps(original)
            async def wrapped(platform_self, session, message_chain, *args, **kwargs):
                chain = _normalize_message(message_chain)
                if chain is not None:
                    if hook._filter_chain(
                        chain,
                        source=f"{cls.__name__}.send_by_session",
                    ):
                        return None
                    message_chain = chain
                return await original(
                    platform_self,
                    session,
                    message_chain,
                    *args,
                    **kwargs,
                )

            return wrapped

        self._patch(cls, "send_by_session", factory)

    def _filter_chain(self, chain: MessageChain, source: str) -> bool:
        if not chain.chain:
            return False

        before_count = len(chain.chain)
        before_plain = collect_visible_text(chain.chain)
        clean_report = sanitize_chain(chain.chain, self.config.clean)
        replace_changes = replace_in_chain(chain.chain, self.config.replace)
        plain = collect_visible_text(chain.chain)

        if clean_report.has_removed():
            logger.debug(
                f"[outputpro:{source}] cleaned outbound chain {before_count}->{len(chain.chain)}: "
                f"{before_plain!r} -> {plain!r}"
            )

        if replace_changes:
            logger.debug(
                f"[outputpro:{source}] replaced outbound text: "
                f"{before_plain!r} -> {plain!r}"
            )

        blocked_word = self._match_block_word(plain)
        if blocked_word:
            logger.info(
                f"[outputpro:{source}] blocked outbound text by word: {blocked_word}"
            )
            chain.chain.clear()
            return True

        return not chain.chain

    def _match_block_word(self, plain: str) -> str | None:
        if not plain:
            return None

        for word in self.config.block.block_words:
            if word and word in plain:
                return word
        return None
