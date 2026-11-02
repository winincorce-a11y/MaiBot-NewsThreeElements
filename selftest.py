"""离线自检：不启动 MaiBot，直接验证插件能否加载、组件声明是否正确、命令与回退文案是否可用。"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

PLUGIN_DIR = Path(__file__).resolve().parent


def load_plugin_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "obituary_newsroom_plugin",
        str(PLUGIN_DIR / "plugin.py"),
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeSend:
    def __init__(self, fail: bool = False) -> None:
        self.texts: List[str] = []
        self.forwards: List[Any] = []
        self.kwargs: List[Dict[str, Any]] = []
        self.fail = fail

    async def text(self, text: str, stream_id: str, **kwargs: Any) -> bool:
        self.kwargs.append(dict(kwargs))
        if self.fail:
            return False
        self.texts.append(text)
        return True

    async def forward(self, messages: List[Dict[str, Any]], stream_id: str, **kwargs: Any) -> bool:
        if self.fail:
            return False
        self.forwards.append(messages)
        return True


class FakeLLM:
    def __init__(self, response: str = "", fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.prompts: List[str] = []
        self.calls: List[Dict[str, Any]] = []

    async def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.prompts.append(prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False))
        self.calls.append(dict(kwargs))
        if self.fail:
            raise RuntimeError("model offline")
        return {"success": True, "response": self.response, "model": "fake"}


class FakeMessage:
    def __init__(self, record: Dict[str, Any] | None = None) -> None:
        self.record = record

    async def get_by_id(self, message_id: str, chat_id: str = "", **kwargs: Any) -> Dict[str, Any]:
        return self.record or {}


class FakeCtx(SimpleNamespace):
    pass


def build_ctx(llm: Any, message_record: Dict[str, Any] | None = None, flag: bool = False) -> FakeCtx:
    send = FakeSend()
    import tempfile
    _tmp = Path(tempfile.mkdtemp(prefix="obit-test-"))
    ctx = FakeCtx(
        send=send, llm=llm, message=FakeMessage(message_record), send_fail=flag,
        paths=SimpleNamespace(data_dir=_tmp),
    )
    if flag:
        send.fail = True
    ctx.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    return ctx


def make_plugin(module: Any, ctx: FakeCtx) -> Any:
    plugin = module.create_plugin()
    plugin._set_context(ctx)
    plugin.set_plugin_config(plugin.get_default_config())
    return plugin



def sent_body(ctx: FakeCtx) -> str:
    """取出实际发出的正文（兼容 text 与 forward 两种发送方式）。"""

    if ctx.send.forwards:
        return ctx.send.forwards[-1][0]["segments"][0]["content"]
    if ctx.send.texts:
        return ctx.send.texts[-1]
    return ""


def sent_all(ctx: FakeCtx) -> List[str]:
    """按发生顺序返回所有发出的正文。"""

    merged: List[tuple[int, str]] = []
    for i, t in enumerate(ctx.send.texts):
        merged.append((i, t))
    for i, fw in enumerate(ctx.send.forwards):
        merged.append((i, fw[0]["segments"][0]["content"]))
    return [t for _, t in merged]


def runner_apply_hook(raw_result: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """复刻宿主 Runner 的 Hook 返回值解析，避免"自己测自己"。

    宿主只认 ``action`` / ``modified_kwargs`` / ``custom_result``；
    返回 ``modified_message`` 这类键会被**静默忽略**（踩过这个坑）。
    """

    if raw_result is None:
        return dict(kwargs)
    if not isinstance(raw_result, dict):
        return dict(kwargs)
    modified = raw_result.get("modified_kwargs")
    if isinstance(modified, dict):
        merged = dict(kwargs)
        merged.update(modified)
        return merged
    return dict(kwargs)

def article_ok(module: Any, plugin: Any, text: str) -> List[str]:
    problems: List[str] = []
    if not text.strip():
        problems.append("正文为空")
    for surname in ("巴兰尼科夫", "尤丁采夫", "布亚诺夫"):
        if surname not in text:
            problems.append(f"缺少当事人：{surname}")
    if "#" in text or "**" in text:
        problems.append("残留 Markdown")
    return problems


def assert_one_paragraph(text: str) -> None:
    """统一断言：输出必须是一段话。"""

    assert text.strip(), "正文为空"
    assert "\n" not in text.strip(), f"应为一整段，实际含换行：{text[:80]!r}"
    assert not text.startswith("【虚构】"), "不应再加【虚构】前缀"
    assert "纯属玩梗" not in text, "不应再加免责声明"


def assert_pivot_structure(
    text: str,
    pivot: str = "值得注意的是",
    names: List[str] | None = None,
) -> None:
    """断言「前半截素材延伸 — 转折词 — 后半截死亡名单」的结构。

    Args:
        text: 待检查正文。
        pivot: 转折词。
        names: 期望出现在后半截的全名；默认沿用插件默认三人。
    """

    assert pivot in text, f"缺少转折词「{pivot}」：{text[:80]!r}"
    head, _, tail = text.partition(pivot)
    assert head.strip(), "转折词前面没有内容"
    assert len(head.strip()) >= 30, f"前半截太短（{len(head.strip())} 字）"
    assert len(tail.strip()) >= 30, f"后半截太短（{len(tail.strip())} 字）"
    assert head.rstrip()[-1] in "。！？；…", f"转折词前应以句号/分号收住，实际是「{head.rstrip()[-1]}」"
    for word in ("身亡", "离世", "死亡", "去世", "不治", "遇难"):
        assert word not in head, f"转折词前不应出现死讯词「{word}」：{head[:60]!r}"
    for name in (names or ["巴兰尼科夫", "尤丁采夫", "布亚诺夫"]):
        assert name in tail, f"当事人 {name} 没有写在转折词之后"


async def main() -> int:
    module = load_plugin_module()

    # ── 1. 组件声明 ──
    probe = make_plugin(module, build_ctx(FakeLLM("x")))
    components = probe.get_components()
    names = {(c["type"], c["name"]) for c in components}
    print("[1] 组件声明：", sorted(names))
    required = {
        ("COMMAND", "news_three_elements"),
        ("TOOL", "write_obituary_news"),
        ("API", "generate_obituary_news"),
        ("HOME_CARD", "obituary_newsroom_card"),
    }
    missing = required - names
    assert not missing, f"缺少组件声明: {missing}"

    command = next(c for c in components if c["name"] == "news_three_elements")
    pattern = command["metadata"]["command_pattern"]
    compiled = re.compile(pattern)
    for sample in ("/新闻三要素 昨天服务器炸了", "/新闻三要素 更新公告", "／新闻三要素 全角斜杠", "/新闻三要素"):
        assert compiled.search(sample), f"命令正则未命中: {sample}"
    # 引用消息时客户端常在命令前插入 @机器人，必须能匹配（否则表现为“毫无反应”）
    for ated in (
        "@麦麦 /新闻三要素",
        "@麦麦 /新闻三要素 美国大选",
        "@麦麦@小号 /新闻三要素 素材",
        "  @麦麦  /新闻三要素 素材",
        "@麦麦／新闻三要素 素材",
    ):
        m_ated = compiled.search(ated)
        assert m_ated, f"带 @ 前缀的命令未命中: {ated}"
        assert m_ated.group("material") in ("", "美国大选", "素材"), f"素材解析错误: {ated}"
    for stale in ("/讣告 素材", "/新闻 素材", "/news 素材", "/obit 素材", "普通聊天 新闻三要素",
                  "新闻三要素", "随便聊聊 /新闻三要素"):
        assert not compiled.search(stale), f"命令正则误命中: {stale}"
    assert not command["metadata"].get("aliases"), "本插件不应再声明任何别名命令"
    print("    命令正则 OK（仅 /新闻三要素）；timeout_ms =", command["metadata"].get("timeout_ms"))

    # ── 2. 配置模型 ──
    default_config = probe.get_default_config()
    schema = probe.get_webui_config_schema(plugin_id="local.obituary-newsroom", plugin_name="讣告新闻社")
    cfg_dump = json.dumps(default_config, ensure_ascii=False)
    assert "巴兰尼科夫" in cfg_dump and "尤丁采夫" in cfg_dump and "布亚诺夫" in cfg_dump
    assert isinstance(default_config["people"]["list"], list), "people.list 应为列表"
    assert len(default_config["people"]["list"]) == 3, "默认三人"
    sections = list(schema.get("sections", {}).keys()) or list(schema.keys())
    print("[2] 配置节：", sections)

    # ── 3. 命令：模型正常 ──
    # 走的是"一段话"形态：不分段、不带标题前缀、不带声明。
    good_article = (
        "据当地通报，事故发生后救援力量赶赴现场处置，周边群众已疏散至安全区域。"
        "有关部门表示，现场处置与原因核查仍在推进，损失情况有待进一步统计。"
        "值得注意的是，现场负责人维亚切斯拉夫·巴兰尼科夫在抢险中被倒塌构件击中身亡；"
        "值班主管安东·尤丁采夫在连续处置后突发心脏骤停离世；"
        "随行工作人员尼基塔·布亚诺夫在巡查泄漏点时中毒身亡。"
        "相关部门表示，事发经过与具体原因仍在进一步调查中。"
    )
    llm = FakeLLM(good_article)
    ctx = build_ctx(llm)
    plugin = make_plugin(module, ctx)
    ok, response, intercept = await plugin.handle_news_command(
        stream_id="stream-1",
        text="/新闻三要素 昨天服务器炸了，修了一夜",
        platform="qq",
        matched_groups={"material": "昨天服务器炸了，修了一夜"},
        message={},
        user_id="10001",
    )
    assert ok, f"命令执行失败: {response}"
    assert intercept is True, "命中的命令应拦截消息，避免麦麦再复述一遍"
    assert sent_all(ctx), "没有发送任何文本"
    sent = sent_body(ctx)
    problems = article_ok(module, plugin, sent)
    assert not problems, f"正文不达标: {problems}"
    assert not sent.startswith("【虚构】"), "不应再加【虚构】前缀"
    assert "虚构创作" not in sent and "纯属玩梗" not in sent, "不应再加免责声明"
    assert "\n" not in sent.strip(), f"应该是一段话，实际含换行：{sent!r}"
    assert len(sent) <= 400, f"太长，应 ≤400 字，实际 {len(sent)}"
    assert_pivot_structure(sent)
    assert "昨天服务器炸了" in llm.prompts[0], "素材没有进入提示词"
    print("[3] 命令输出 OK（单段 / 无前缀 / 无声明 / 前-转折-后 结构），字数 =", len(sent))
    print("------ 输出示例 ------")
    print(sent)
    print("----------------------")

    # ── 4. 冷却 ──
    ok2, response2, _ = await plugin.handle_news_command(
        stream_id="stream-1", text="/新闻三要素 再来一次", platform="qq",
        matched_groups={"material": "再来一次"}, message={},
    )
    assert ok2 and "秒后再试" in ctx.send.texts[-1], f"冷却未生效: {response2}"
    print("[4] 冷却 OK")

    # ── 5. 模型失败 → 本地模板 ──
    ctx_fail = build_ctx(FakeLLM(fail=True))
    plugin_fail = make_plugin(module, ctx_fail)
    ok3, response3, _ = await plugin_fail.handle_news_command(
        stream_id="stream-2", text="/新闻三要素 版本更新推迟", platform="qq",
        matched_groups={"material": "版本更新推迟"}, message={},
    )
    assert ok3, f"回退失败: {response3}"
    fallback_text = sent_body(ctx_fail)
    problems = article_ok(module, plugin_fail, fallback_text)
    assert not problems, f"回退正文不达标: {problems}"
    assert "版本更新推迟" in fallback_text or "近期的一项工作" in fallback_text
    print("[5] 本地模板回退 OK，字数 =", len(fallback_text))

    # ── 6. 模型漏人 → 重写 → 补齐 ──
    partial = "短讯：资深制作人维亚切斯拉夫·巴兰尼科夫离世。\n相关情况仍在调查中。"
    full = good_article
    llm_repair = FakeLLM(partial)
    original_generate = llm_repair.generate
    counter = {"n": 0}

    async def generate_side_effect(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        counter["n"] += 1
        if counter["n"] == 1:
            return await original_generate(prompt, **kwargs)
        llm_repair.response = full
        return await original_generate(prompt, **kwargs)

    llm_repair.generate = generate_side_effect  # type: ignore[assignment]
    ctx_repair = build_ctx(llm_repair)
    plugin_repair = make_plugin(module, ctx_repair)
    await plugin_repair.handle_news_command(
        stream_id="stream-3", text="/新闻三要素 素材", platform="qq",
        matched_groups={"material": "素材"}, message={},
    )
    repaired = sent_body(ctx_repair)
    problems = article_ok(module, plugin_repair, repaired)
    assert not problems, f"补写失败: {problems}"
    assert counter["n"] >= 2, "没有触发补写"
    print("[6] 漏人自动补写 OK，调用次数 =", counter["n"])

    # ── 6b. 回归：provider 把思考链当正文返回（线上真实故障） ──
    reasoning_leak = (
        "【虚构】我们需要回答中文新闻通稿，完全虚构玩梗。需要符合要求。素材只有\"美国和伊朗发生战争\"。"
        "需要写短篇新闻，约700字，上限1800。必须融入三人死讯，每人全名一次，后文指代。"
        "死因贴合场景：战争、行业？三人是游戏公司？巴兰尼科夫、尤丁采夫、布亚诺夫是 Gaijin？资深制作人，CEO，运营总监。"
        "素材美国和伊朗发生战争。需要虚构新闻，严肃克制。能否报道战争爆发对某游戏公司运营影响？需要自然融入。"
        "不能编造可核查具体细节。需要保留关键事实。需要可能写\"当地时间X日\"。"
        "Need avoid making it sound like obituary list. Use internal notice, two unnamed colleagues. "
        "Need direct output. "
        "Need about 700 Chinese chars. The names are real? We can refer to real people? Use as given.\n\n"
        "标题：美伊冲突波及游戏服务　一互动娱乐企业三名高管及制作人相继离世\n\n"
        "本报讯 据……报道。"
    )

    class LeakLLM:
        """总是返回思考链，用来验证插件不会把它当正文发出去。"""

        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
            self.calls += 1
            return {"success": True, "response": reasoning_leak, "reasoning": reasoning_leak, "model": "fake"}

    leak_llm = LeakLLM()
    ctx_leak = build_ctx(leak_llm)
    plugin_leak = make_plugin(module, ctx_leak)
    ok_leak, _, _ = await plugin_leak.handle_news_command(
        stream_id="stream-leak", text="/新闻三要素 美国和伊朗发生战争", platform="qq",
        matched_groups={"material": "美国和伊朗发生战争"}, message={},
    )
    assert ok_leak, "思考链场景下命令整体仍应成功（走模板兜底）"
    leak_out = sent_body(ctx_leak)
    assert "我们需要回答" not in leak_out, "思考链被当成正文发出去了"
    assert "Need avoid" not in leak_out, "思考链里的英文元指令漏进正文"
    assert "标题：" not in leak_out, "提纲式标记漏进正文"
    problems = article_ok(module, plugin_leak, leak_out)
    assert not problems, f"思考链兜底后正文不达标: {problems}"
    assert leak_llm.calls >= 1, "应当尝试过调用模型"
    print(f"[6b] 思考链防护 OK（模型调用 {leak_llm.calls} 次，最终走合格输出）")

    # ── 6c. response 与 reasoning 完全相同时直接丢弃该结果 ──
    class DupLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
            self.calls += 1
            same = "这是一段被误当作正文的思考内容，需要再想想。"
            return {"success": True, "response": same, "reasoning": same}

    dup_llm = DupLLM()
    ctx_dup = build_ctx(dup_llm)
    plugin_dup = make_plugin(module, ctx_dup)
    await plugin_dup.handle_news_command(
        stream_id="stream-dup", text="/新闻三要素 素材", platform="qq",
        matched_groups={"material": "素材"}, message={},
    )
    dup_out = sent_body(ctx_dup)
    assert "被误当作正文" not in dup_out, "response==reasoning 的结果没被丢弃"
    assert "巴兰尼科夫" in dup_out, "应回退到模板"
    print("[6c] response==reasoning 丢弃 OK，调用次数 =", dup_llm.calls)

    # ── 6d. 回归：set_reply 把正文静默吞掉（线上真实故障） ──
    # 宿主在 set_reply=True 且无 reply_message 时直接 return None，正文一个字都发不出去。
    ctx_sr = build_ctx(FakeLLM(good_article))
    plugin_sr = make_plugin(module, ctx_sr)
    await plugin_sr.handle_news_command(
        stream_id="stream-sr", text="/新闻三要素 素材", platform="qq",
        matched_groups={"material": "素材"}, message={},
    )
    assert sent_all(ctx_sr), "正文没有发出"
    for kw in ctx_sr.send.kwargs:
        assert not kw.get("set_reply"), f"插件不应传 set_reply（会让宿主丢弃整条消息）：{kw}"
    assert plugin_sr.config.output.set_reply is False, "output.set_reply 默认必须是 false"
    print("[6d] set_reply 默认关闭且从不传给宿主 OK")

    # ── 6e. 发送失败时不能进入冷却，否则重试会被"还在排字"挡住 ──
    class CantSendLLM(FakeLLM):
        pass

    ctx_cs = build_ctx(CantSendLLM(good_article), flag=True)
    plugin_cs = make_plugin(module, ctx_cs)
    ok_cs, reason_cs, _ = await plugin_cs.handle_news_command(
        stream_id="stream-cs", text="/新闻三要素 素材", platform="qq",
        matched_groups={"material": "素材"}, message={},
    )
    assert not ok_cs and "发送失败" in reason_cs, f"发送失败未被感知: {reason_cs}"
    assert plugin_cs._cooldown_remaining("stream-cs") == 0, "发送失败后不应记冷却"
    print("[6e] 发送失败可感知且不吞掉重试 OK")
    # ── 7. 引用消息取素材 ──
    ctx_quote = build_ctx(FakeLLM(good_article), {"processed_plain_text": "引用里的素材：服务器迁移失败"})
    plugin_quote = make_plugin(module, ctx_quote)
    ok7, _, _ = await plugin_quote.handle_news_command(
        stream_id="stream-4", text="/新闻三要素", platform="qq",
        matched_groups={"material": ""}, message={"reply_to": "msg-1"},
    )
    assert ok7
    assert "服务器迁移失败" in ctx_quote.llm.prompts[0], "引用素材没有进入提示词"
    print("[7] 引用消息取素材 OK")

    # ── 8. 帮助与平台过滤 ──
    ctx_help = build_ctx(FakeLLM(good_article))
    plugin_help = make_plugin(module, ctx_help)
    await plugin_help.handle_news_command(
        stream_id="stream-5", text="/新闻三要素", platform="qq", matched_groups={"material": ""}, message={},
    )
    assert "使用说明" in ctx_help.send.texts[-1]
    config = plugin_help.get_default_config()
    config["plugin"]["only_platforms"] = ["telegram"]
    plugin_help.set_plugin_config(config)
    before = len(ctx_help.send.texts)
    ok8, reason8, intercept8 = await plugin_help.handle_news_command(
        stream_id="stream-5", text="/新闻三要素 素材", platform="qq",
        matched_groups={"material": "素材"}, message={},
    )
    # 第三项是「拦截消息」标志：前置拦截不应拦截，命令要落回普通聊天流程。
    assert not ok8 and "平台" in reason8 and not intercept8
    assert len(ctx_help.send.texts) == before, "平台不匹配时不应发送任何消息"
    print("[8] 帮助与平台过滤 OK")

    # ── 8b. 「重写」沿用上一次素材 ──
    config["plugin"]["only_platforms"] = []
    config["plugin"]["cooldown_seconds"] = 0
    plugin_help.set_plugin_config(config)
    await plugin_help.handle_news_command(
        stream_id="stream-r", text="/新闻三要素 一手素材", platform="qq",
        matched_groups={"material": "一手素材"}, message={},
    )
    await plugin_help.handle_news_command(
        stream_id="stream-r", text="/新闻三要素 重写", platform="qq",
        matched_groups={"material": "重写"}, message={},
    )
    assert "一手素材" in ctx_help.llm.prompts[-1], "重写没有复用上一次素材"
    ctx_empty = build_ctx(FakeLLM(good_article))
    plugin_empty = make_plugin(module, ctx_empty)
    await plugin_empty.handle_news_command(
        stream_id="stream-none", text="/新闻三要素 重写", platform="qq",
        matched_groups={"material": "重写"}, message={},
    )
    assert "还没出过稿" in ctx_empty.send.texts[-1]
    print("[8b] 重写命令 OK")

    # ── 9. Tool 与 API ──
    ctx_tool = build_ctx(FakeLLM(good_article))
    plugin_tool = make_plugin(module, ctx_tool)
    tool_result = await plugin_tool.handle_tool_write_news(stream_id="stream-6", material="工具调用素材")
    assert tool_result["success"] and "巴兰尼科夫" in tool_result["content"]
    assert sent_all(ctx_tool), "Tool 没有发送消息"
    api_result = await plugin_tool.api_generate_news(material="接口调用素材", stream_id="stream-7")
    assert api_result["success"] and api_result["text"]
    try:
        await plugin_tool.api_generate_news(material="   ")
    except ValueError:
        pass
    else:
        raise AssertionError("空素材应抛 ValueError")
    print("[9] Tool / API OK")

    # ── 10. 超长输入被压成一段并截断（不再走合并转发） ──
    long_article = good_article + ("另据通报，相关情况仍在进一步调查中。" * 200)
    ctx_long = build_ctx(FakeLLM(long_article))
    plugin_long = make_plugin(module, ctx_long)
    await plugin_long.handle_news_command(
        stream_id="stream-8", text="/新闻三要素 长素材", platform="qq",
        matched_groups={"material": "长素材"}, message={},
    )
    assert sent_all(ctx_long), "超长输入应当直接发一段文本"
    long_out = sent_body(ctx_long)
    assert_one_paragraph(long_out)
    assert len(long_out) <= 400, f"未截断到 400 字以内: {len(long_out)}"
    assert long_out.endswith("🙏"), "截断后应保住哀悼结尾"
    print("[10] 超长输入压成一段 + 截断 OK，长度 =", len(long_out))

    # ── 11. 身份随新闻场景变化（而非固定的制作人/CEO） ──
    prompt_war = plugin._build_prompt("美国和伊朗发生战争")
    assert "资深制作人" not in prompt_war, "提示词里不应再出现写死的现实职务"
    assert "首席执行官" not in prompt_war, "提示词里不应再出现写死的现实职务"
    assert "军事" in prompt_war and "指挥官" in prompt_war, "提示词应给出按场景设定身份的指引"
    assert "只有一段" in prompt_war or "一段" in prompt_war, "提示词应要求单段"

    # 回退模板也要按场景换身份
    roles_war, causes_war = plugin._scene_roles("美国和伊朗发生战争，边境交火")
    assert "指挥官" in roles_war[0], f"战争素材应映射到军官身份，实际 {roles_war}"
    roles_sport, _ = plugin._scene_roles("昨晚的联赛比赛")
    assert "教练" in roles_sport[0], f"体育素材应映射到教练身份，实际 {roles_sport}"
    roles_factory, _ = plugin._scene_roles("化工厂车间发生爆炸泄漏")
    assert "主管" in roles_factory[0] or "班长" in roles_factory[0], f"事故素材映射异常 {roles_factory}"
    assert len(causes_war) >= 3, f"死因池应至少有 3 条，实际 {len(causes_war)}"
    assert len(set(causes_war)) == len(causes_war), "死因池内不应重复"
    assert len(roles_war) == len(causes_war), "身份池与死因池应等长"

    fallback = plugin._fallback_news("美国和伊朗发生战争，边境交火")
    assert_one_paragraph(fallback)
    assert_pivot_structure(fallback)
    for bad in ("制作人", "首席执行官", "运营总监"):
        assert bad not in fallback, f"回退模板不应再出现现实职务：{bad}"
    print("[11] 身份随场景变化 OK：战争→", roles_war, "｜体育→", roles_sport)


    # ── 12. 自定义死亡名单（换人 / 增人） ──
    custom_cfg = plugin.get_default_config()
    custom_cfg["people"]["list"] = [
        {"name": "张三", "title": "", "cause_hint": ""},
        {"name": "李四·王五", "title": "", "cause_hint": ""},
    ]
    ctx_custom = build_ctx(FakeLLM(good_article))
    plugin_custom = make_plugin(module, ctx_custom)
    plugin_custom.set_plugin_config(custom_cfg)
    names_custom = plugin_custom._people_names()
    assert names_custom == ["张三", "李四·王五"], f"自定义名单未生效：{names_custom}"
    prompt_custom = plugin_custom._build_prompt("测试素材")
    assert "张三" in prompt_custom and "李四·王五" in prompt_custom
    assert "巴兰尼科夫" not in prompt_custom, "换人后不应再出现默认三人"
    assert "共 **2 人**" in prompt_custom, "人数应随名单变化"
    print("[12] 自定义名单（2 人）OK：", names_custom)

    # ── 13. 五人名单 + 回退模板也要写全 ──
    five_cfg = plugin.get_default_config()
    five_cfg["people"]["list"] = [
        {"name": "甲一", "title": "", "cause_hint": ""},
        {"name": "乙二", "title": "", "cause_hint": ""},
        {"name": "丙三", "title": "", "cause_hint": ""},
        {"name": "丁四", "title": "", "cause_hint": ""},
        {"name": "戊五", "title": "", "cause_hint": ""},
    ]
    plugin_five = make_plugin(module, build_ctx(FakeLLM("")))
    plugin_five.set_plugin_config(five_cfg)
    assert len(plugin_five._people()) == 5
    fallback5 = plugin_five._fallback_news("美国和伊朗发生战争")
    for nm in ("甲一", "乙二", "丙三", "丁四", "戊五"):
        assert nm in fallback5, f"回退模板漏写 {nm}"
    assert_one_paragraph(fallback5)
    assert_pivot_structure(fallback5, names=["甲一", "乙二", "丙三", "丁四", "戊五"])
    print("[13] 五人名单回退模板 OK，字数 =", len(fallback5))

    # ── 14. max_people 上限生效 ──
    cap_cfg = plugin.get_default_config()
    cap_cfg["people"]["list"] = [{"name": f"某{i}", "title": "", "cause_hint": ""} for i in range(1, 11)]
    cap_cfg["people"]["max_people"] = 4
    plugin_cap = make_plugin(module, build_ctx(FakeLLM("")))
    plugin_cap.set_plugin_config(cap_cfg)
    assert len(plugin_cap._people()) == 4, f"上限未生效：{len(plugin_cap._people())}"
    assert plugin_cap._people_names() == ["某1", "某2", "某3", "某4"]
    print("[14] max_people 上限 OK：10 条名单截断为 4 人")

    # ── 15. 换人后漏人校验跟着走 ──
    text_only_zhang = "据通报，当地发生一起事故。值得注意的是，张三在处置中身亡，相关情况仍在调查中。"
    problems15 = plugin_custom._judge(text_only_zhang)
    assert any("李四·王五" in p for p in problems15), f"应检出漏掉自定义当事人：{problems15}"
    assert not any("巴兰尼科夫" in p for p in problems15), "不应再校验默认三人"
    print("[15] 换人后的漏人校验 OK")


    # ── 16. 括号里的名单被正确解析（全角/半角/顿号都认） ──
    split_cases = [
        ("美国大选 （特朗普，奥巴马，拜登）", "美国大选", ["特朗普", "奥巴马", "拜登"]),
        ("美国大选 (特朗普, 奥巴马, 拜登)", "美国大选", ["特朗普", "奥巴马", "拜登"]),
        ("美国大选（特朗普、奥巴马、拜登）", "美国大选", ["特朗普", "奥巴马", "拜登"]),
        ("昨晚服务器崩了，回滚三次 （张三，李四，王五）",
         "昨晚服务器崩了，回滚三次", ["张三", "李四", "王五"]),
        ("某公司周三发布财报（营收下滑）（张三，李四）",
         "某公司周三发布财报（营收下滑）", ["张三", "李四"]),
        ("美国大选", "美国大选", []),
        ("美国大选 （）", "美国大选 （）", []),
        ("今早高铁脱轨 温斯顿·丘吉尔 富兰克林·罗斯福",
         "今早高铁脱轨 温斯顿·丘吉尔 富兰克林·罗斯福", []),
    ]
    for raw, want_material, want_names in split_cases:
        got_material, got_names = plugin._split_material_and_names(raw)
        assert got_material == want_material, f"素材解析错误：{raw!r} -> {got_material!r}"
        assert got_names == want_names, f"人名解析错误：{raw!r} -> {got_names}"
    print("[16] 括号名单解析 OK（8 组用例，含全角/半角/顿号/空括号/素材自带括号）")

    # ── 17. 命令带人名时，新闻里换成这些人 ──
    class TwoNameLLM:
        """第一次返回含张三李四的正文，验证名单真的透传到了生成与校验。"""

        def __init__(self) -> None:
            self.prompts: List[str] = []

        async def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
            self.prompts.append(prompt)
            return {
                "success": True,
                "response": (
                    "某地发生一起事故，有关部门已到场处置，现场秩序正在恢复。"
                    "值得注意的是，据当地通报，张三在抢险中身亡；"
                    "另一名工作人员李四在赶赴现场途中遭遇交通事故身亡。"
                    "相关情况仍在调查中。"
                ),
                "reasoning": "",
            }

    llm17 = TwoNameLLM()
    ctx17 = build_ctx(llm17)
    plugin17 = make_plugin(module, ctx17)
    ok17, _, _ = await plugin17.handle_news_command(
        stream_id="stream-n17", text="/新闻三要素 某地发生事故（张三，李四）", platform="qq",
        matched_groups={"material": "某地发生事故（张三，李四）"}, message={},
    )
    assert ok17, "带人名的命令执行失败"
    out17 = sent_body(ctx17)
    assert "张三" in out17 and "李四" in out17, f"自定义人名没进正文：{out17[:80]}"
    assert "巴兰尼科夫" not in out17, "不应再出现默认三人"
    assert "张三" in llm17.prompts[0] and "李四" in llm17.prompts[0], "人名没进提示词"
    assert "巴兰尼科夫" not in llm17.prompts[0], "提示词里仍有默认三人"
    assert_pivot_structure(out17, names=["张三", "李四"])
    print("[17] 命令指定人名生效 OK：", out17[:40], "…")

    # ── 18. 不带人名时用默认三人 ──
    ctx18 = build_ctx(FakeLLM(
        "某地发生事故，有关部门已到场处置。值得注意的是，据当地通报，"
        "维亚切斯拉夫·巴兰尼科夫在抢险中身亡；安东·尤丁采夫在途中遭遇车祸身亡；"
        "尼基塔·布亚诺夫突发疾病离世。相关情况仍在调查中。"
    ))
    plugin18 = make_plugin(module, ctx18)
    await plugin18.handle_news_command(
        stream_id="stream-n18", text="/新闻三要素 某地发生事故", platform="qq",
        matched_groups={"material": "某地发生事故"}, message={},
    )
    out18 = sent_body(ctx18)
    assert all(n in out18 for n in ("巴兰尼科夫", "尤丁采夫", "布亚诺夫")), "默认三人缺失"
    assert "张三" not in out18
    print("[18] 不带人名时沿用默认三人 OK")


    # ── 19. 结尾句：去掉"仍在调查"套话，换成"让我们为他们X🙏" ──
    tail_samples = [
        "某地发生事故，救援已展开。值得注意的是，据通报，张三在抢险中身亡，李四在处置后离世。相关情况仍在调查中。",
        "某地发生事故，救援已展开。值得注意的是，据通报，张三在抢险中身亡。具体原因尚待进一步核实。",
        "某地发生事故，救援已展开。值得注意的是，据通报，张三在抢险中身亡。目前，相关情况仍在进一步调查中。",
        "某地发生事故，救援已展开。值得注意的是，据通报，张三在抢险中身亡。",
    ]
    for sample in tail_samples:
        finalized = plugin._finalize(sample)
        assert finalized.endswith("🙏"), f"结尾没有补哀悼句：{finalized[-30:]!r}"
        assert "让我们为他们" in finalized, f"结尾模板不对：{finalized[-30:]!r}"
        for word in ("仍在调查", "仍待进一步核实", "尚待进一步核实", "仍在进一步调查"):
            assert word not in finalized, f"套话没剥干净（{word}）：{finalized[-40:]!r}"
    assert "。让我们为他们" in plugin._finalize(tail_samples[0]), "结尾句前应有句号"
    print("[19] 结尾句替换 + 套话剥离 OK（4 组用例）")

    # ── 20. 自定义词库随机填充 ──
    from collections import Counter

    verbs = ["作哀", "悼观", "设敬", "应敬", "应悼"]
    seen20 = Counter()
    for _ in range(120):
        seen20[plugin._pick_verb()] += 1
    assert set(seen20) == set(verbs), f"随机词覆盖不全：{set(seen20)}"
    assert all(cnt > 0 for cnt in seen20.values()), "每个词都应被抽到过"
    print("[20] 随机词分布 OK：", dict(seen20))

    # ── 21. 用户自定义词库与模板 ──
    cfg21 = plugin.get_default_config()
    cfg21["ending"]["verbs"] = ["超度", "送别"]
    cfg21["ending"]["template"] = "愿他们{verb}"
    cfg21["ending"]["use_emoji"] = False
    plugin21 = make_plugin(module, build_ctx(FakeLLM("")))
    plugin21.set_plugin_config(cfg21)
    got21 = {plugin21._render_ending() for _ in range(40)}
    assert got21 == {"愿他们超度", "愿他们送别"}, f"自定义模板/词库未生效：{got21}"
    print("[21] 自定义词库与模板 OK：", got21)

    # 只留一个词时固定用它
    cfg22 = plugin.get_default_config()
    cfg22["ending"]["verbs"] = ["致敬"]
    plugin22 = make_plugin(module, build_ctx(FakeLLM("")))
    plugin22.set_plugin_config(cfg22)
    assert plugin22._render_ending() == "让我们为他们致敬🙏"
    # 词库为空时回落「作哀」
    cfg23 = plugin.get_default_config()
    cfg23["ending"]["verbs"] = []
    plugin23 = make_plugin(module, build_ctx(FakeLLM("")))
    plugin23.set_plugin_config(cfg23)
    assert plugin23._render_ending() == "让我们为他们作哀🙏"
    print("[22] 单词固定 / 空词库回落 OK")

    # ── 23. 关闭 ending 时不改结尾 ──
    cfg24 = plugin.get_default_config()
    cfg24["ending"]["enabled"] = False
    plugin24 = make_plugin(module, build_ctx(FakeLLM("")))
    plugin24.set_plugin_config(cfg24)
    raw24 = "某地发生事故，救援已展开。值得注意的是，张三在抢险中身亡。相关情况仍在调查中。"
    assert plugin24._finalize(raw24) == raw24, "关闭 ending 后不应改动结尾"
    print("[23] 关闭 ending 时保留原结尾 OK")


    # ── 24. 开头署名开关（默认关闭） ──
    body24 = "美国大选进入计票阶段，多地加强安保。值得注意的是，据通报，张三在值守后突发疾病离世。"
    default24 = plugin._finalize(body24)
    assert not default24.startswith("据"), f"开头署名默认应关闭：{default24[:20]!r}"
    assert "盖金社" not in default24, "默认不应出现盖金社"
    print("[24] 开头署名默认关闭 OK")

    cfg25 = plugin.get_default_config()
    cfg25["output"]["lead_in"] = True
    plugin25 = make_plugin(module, build_ctx(FakeLLM("")))
    plugin25.set_plugin_config(cfg25)
    out25 = plugin25._finalize(body24)
    assert out25.startswith("据盖金社报道，"), f"开启后应以「据盖金社报道，」开头：{out25[:20]!r}"

    # 自定义机构名
    cfg26 = plugin.get_default_config()
    cfg26["output"]["lead_in"] = True
    cfg26["output"]["agency"] = "路透社"
    plugin26 = make_plugin(module, build_ctx(FakeLLM("")))
    plugin26.set_plugin_config(cfg26)
    out26 = plugin26._finalize(body24)
    assert out26.startswith("据路透社报道，"), f"自定义机构未生效：{out26[:20]!r}"
    print("[25] 开头署名开启 / 自定义机构 OK：", out26[:22], "…")

    # 模型自己写了电头时要去重，且不能把正文吃掉
    heads = [
        "据当地媒体报道，美国大选进入计票阶段。值得注意的是，张三离世。",
        "本报讯，美国大选进入计票阶段。值得注意的是，张三离世。",
        "本报讯（记者张三），美国大选进入计票阶段。值得注意的是，李四离世。",
        "据美国有线电视新闻网报道，美国大选进入计票阶段。值得注意的是，张三离世。",
        "盖金社11月1日电，美国大选进入计票阶段。值得注意的是，张三离世。",
    ]
    for raw in heads:
        out = plugin25._finalize(raw)
        assert out.count("据盖金社报道，") == 1, f"电头重复：{out[:40]!r}"
        assert out.count("让我们为他们") == 1, f"结尾重复：{out[-30:]!r}"
        assert "计票阶段" in out, f"正文被电头正则吃掉了：{out[:40]!r}"
    print("[26] 电头去重 OK（5 种写法，正文未被误删）")

    # ── 27. 插件显示名 ──
    meta = module.create_plugin()
    schema27 = meta.get_webui_config_schema(plugin_id="local.obituary-newsroom", plugin_name="新闻三要素")
    names27 = [c27["name"] for c27 in meta.get_components() if c27["type"] == "HOME_CARD"]
    assert names27, "首页卡片缺失"
    assert meta.get_default_config()["output"]["forward_nickname"] == "盖金社"
    assert meta._help_text().startswith("📰 新闻三要素"), f"帮助标题不对：{meta._help_text()[:20]!r}"
    assert schema27, "配置 schema 为空"
    print("[27] 显示名 OK：帮助标题=新闻三要素，转发署名=盖金社")


    # ── 28. 素材含死亡事件时改用「替换死者」模式 ──
    death_cases = [
        "今早高铁脱轨，造成3人死亡",
        "昨晚化工厂爆炸，多人遇难",
        "某地发生事故，两名工人不幸身亡",
        "地震导致200人罹难",
    ]
    alive_cases = ["美国大选", "学校篮球队夺冠", "服务器崩了", "昨晚化工厂车间爆炸了"]
    for t in death_cases:
        assert module._material_has_death(t), f"应识别为含死亡：{t}"
    for t in alive_cases:
        assert not module._material_has_death(t), f"不应识别为含死亡：{t}"

    p_death = plugin._build_prompt("今早高铁脱轨，造成3人死亡")
    p_alive = plugin._build_prompt("美国大选")
    assert "已经写了有人死亡" in p_death, "含死亡素材应走替换模式提示词"
    assert "已经写了有人死亡" not in p_alive, "普通素材不应走替换模式"
    assert "值得注意的是" in p_alive, "普通素材仍应要求转折词"
    assert "不要另外虚构一起新的死亡事件" in p_death
    assert "保留素材原本的死因与场景" in p_death
    print("[28] 含死亡素材的识别与提示词切换 OK")

    # ── 29. 替换模式下不要求转折词，且出现转折词会判不合格 ──
    names29 = ["张三", "李四"]
    roster29 = [module.PersonConfig(name=n) for n in names29]
    body_ok = (
        "今早一列高铁发生脱轨，多节车厢偏离线路，救援力量赶赴现场处置。"
        "据当地通报，事故造成两人死亡，分别为现场调度员张三与随车检修员李四。"
    )
    assert plugin._judge_structure(body_ok, roster29, True) == [], "替换模式不应要求转折词"
    body_bad = (
        "今早一列高铁发生脱轨，多节车厢偏离线路。"
        "值得注意的是，据当地通报，张三与李四在事故中身亡。"
    )
    bad_problems = plugin._judge_structure(body_bad, roster29, True)
    assert any("不要再用" in p for p in bad_problems), f"出现转折词应判不合格：{bad_problems}"
    # 同样结构在普通模式下应当合格（前半截要够长）
    body_normal = (
        "今早一列高铁发生脱轨，多节车厢偏离线路，救援力量赶赴现场处置，"
        "区间运输一度中断，伤者已送医救治，脱轨原因正在排查。"
        "值得注意的是，据当地通报，现场调度员张三与随车检修员李四在事故中身亡，"
        "另有一名随车技术员在转运途中引发旧疾，经抢救无效离世。"
    )
    roster29b = [module.PersonConfig(name=n) for n in ("张三", "李四", "王五")]
    body_normal = body_normal.replace("另有一名随车技术员", "随车技术员王五")
    assert plugin._judge_structure(body_normal, roster29b, False) == [], "普通模式下该结构应合格"
    print("[29] 替换模式的结构校验 OK（不要求转折词，出现则判不合格）")

    # ── 30. 替换模式的回退模板也不带转折词 ──
    fb_death = plugin._fallback_news("今早高铁脱轨，造成3人死亡")
    assert "值得注意的是" not in fb_death, f"含死亡素材的回退模板不应带转折词：{fb_death[:40]}"
    assert "已造成人员伤亡" in fb_death
    for nm in ("巴兰尼科夫", "尤丁采夫", "布亚诺夫"):
        assert nm in fb_death, f"回退模板漏写 {nm}"
    fb_alive = plugin._fallback_news("美国大选")
    assert "值得注意的是" in fb_alive, "普通素材的回退模板应保留转折词"
    print("[30] 替换模式回退模板 OK")


    # ── 31. stream_id 过滤（模型会编 "current"） ──
    fake_ids = ["current", "当前会话", "abc", "stream", "  ", "this", "会话"]
    good_ids = ["c58628a91bab2b87117e274bd85f05fa", "A1B2C3D4E5", "c58628a9"]
    for v in fake_ids:
        assert plugin._sanitize_stream_id(v) == "", f"应丢弃假 stream_id: {v!r}"
    for v in good_ids:
        assert plugin._sanitize_stream_id(v) == v.strip(), f"应保留真实 stream_id: {v!r}"
    print("[31] stream_id 过滤 OK（假 ID 全部丢弃，真实 ID 保留）")

    # ── 32. 工具被传 "current" 时回落到最近真实会话 ──
    ctx32 = build_ctx(FakeLLM(
        "某地发生事故，救援力量已到场处置，周边群众疏散至安全区域，现场秩序逐步恢复。"
        "值得注意的是，据通报，维亚切斯拉夫·巴兰尼科夫在一线抢险中身亡；"
        "安东·尤丁采夫在处置过程中离世；尼基塔·布亚诺夫突发疾病不治。"
    ))
    plugin32 = make_plugin(module, ctx32)
    plugin32._last_stream_id = "c58628a91bab2b87117e274bd85f05fa"
    r32 = await plugin32.handle_tool_write_news(stream_id="current", material="某地事故")
    assert r32["success"], f"应回落成功：{r32}"
    assert sent_all(ctx32), "没有发送"
    print("[32] 工具传 current 时回落真实会话 OK")

    # 没有任何可用会话时，明确失败而不是静默
    ctx33 = build_ctx(FakeLLM("x"))
    plugin33 = make_plugin(module, ctx33)
    r33 = await plugin33.handle_tool_write_news(material="某地事故")
    assert not r33["success"] and "会话" in r33["content"], f"应明确说明拿不到会话：{r33}"
    assert not sent_all(ctx33), "没有会话时不应发送"
    print("[33] 无会话时明确失败 OK")

    # ── 34. 引用消息取不到文字时不硬编新闻 ──
    ctx34 = build_ctx(FakeLLM("不应该被调用"), None)
    plugin34 = make_plugin(module, ctx34)
    ok34, _, _ = await plugin34.handle_news_command(**{
        "text": "/新闻三要素", "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq", "matched_groups": {"material": ""},
        "message": {"reply_to": "m1"}, "user_id": "u",
        "is_local_operator": False, "plugin_config": {},
    })
    assert ok34
    assert ctx34.send.texts, "应给出提示"
    assert "引用" in ctx34.send.texts[-1] and "素材" in ctx34.send.texts[-1], \
        f"应提示引用取不到内容：{ctx34.send.texts[-1][:40]!r}"
    print("[34] 引用无文字时给提示而非硬编 OK")


    # ── 35. 工具参数 Schema 合法性（数组必须带 items_schema） ──
    # 这条是真实事故的回归：names 是数组却没给 items_schema，
    # 宿主 normalize_tool_option 直接抛 ValueError，把整个聊天循环带崩。
    tool_comp = next(c35 for c35 in components if c35["type"] == "TOOL")
    params35 = tool_comp["metadata"].get("parameters") or []
    assert params35, "工具应当声明参数"
    for p35 in params35:
        if p35["param_type"] == "array":
            assert p35.get("items_schema"), (
                f"数组参数 {p35['name']} 必须提供 items_schema，否则宿主枚举工具时会抛异常"
            )
        if p35["param_type"] == "object":
            assert p35.get("properties") is not None, (
                f"对象参数 {p35['name']} 必须提供 properties"
            )
    names_param = next((p for p in params35 if p["name"] == "names"), None)
    assert names_param is not None, "应声明 names 参数"
    assert names_param["items_schema"]["type"] == "string", f"names 的 items 应为 string：{names_param}"
    assert names_param["required"] is False, "names 应为可选"
    stream_param = next((p for p in params35 if p["name"] == "stream_id"), None)
    assert stream_param is not None and stream_param["required"] is False, \
        "stream_id 必须可选（宿主工具链不注入它）"
    print("[35] 工具参数 Schema 合法 OK（数组有 items_schema，stream_id 可选）")


    # ── 36. 引用消息的多种形态都要能取到素材 ──
    class QuoteMsg:
        def __init__(self, rec=None): self.rec = rec
        async def get_by_id(self, mid, chat_id="", **kw): return self.rec

    async def quote_case(message, rec=None):
        ctx_q = SimpleNamespace(
            send=FakeSend(), llm=FakeLLM(""), message=QuoteMsg(rec),
            logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                                   error=lambda *a, **k: None, debug=lambda *a, **k: None),
        )
        pl_q = module.create_plugin()
        pl_q._set_context(ctx_q)
        pl_q.set_plugin_config(pl_q.get_default_config())
        return await pl_q._material_from_quote(message, "c58628a91bab2b87117e274bd85f05fa")

    # 形态1：raw_message 里的 reply 段（适配器最常见）
    got1 = await quote_case({
        "raw_message": [
            {"type": "reply", "data": {"target_message_content": "昨晚空调坏了"}},
            {"type": "text", "data": "/新闻三要素"},
        ],
        "processed_plain_text": "/新闻三要素",
    })
    assert got1 == "昨晚空调坏了", f"reply 段提取失败：{got1!r}"

    # 形态2：reply 段只有 ID，靠查库
    got2 = await quote_case(
        {"raw_message": [{"type": "reply", "data": {"target_message_id": "m0"}}]},
        {"processed_plain_text": "库里的引用正文"},
    )
    assert got2 == "库里的引用正文", f"按 ID 查库失败：{got2!r}"

    # 形态3：[引用]…[/引用] 包裹
    got3 = await quote_case({"processed_plain_text": "[引用] 昨晚空调坏了 [/引用] /新闻三要素"})
    assert got3 == "昨晚空调坏了", f"[引用] 段提取失败：{got3!r}"

    # 形态4：顶层 reply_to
    got4 = await quote_case({"reply_to": "m0"}, {"processed_plain_text": "顶层 reply_to 正文"})
    assert got4 == "顶层 reply_to 正文", f"reply_to 查库失败：{got4!r}"

    # 没有任何引用信息 -> 空
    got5 = await quote_case({"processed_plain_text": "/新闻三要素 直接给素材"})
    assert got5 == "", f"无引用时应返回空：{got5!r}"
    print("[36] 引用消息四种形态提取 OK")

    # ── 37. 引用 + 命令 的完整链路能出稿并发出 ──
    article37 = (
        "某地发生事故，救援力量已到场处置，周边群众疏散至安全区域，现场秩序正逐步恢复。"
        "值得注意的是，据通报，维亚切斯拉夫·巴兰尼科夫在一线抢险中身亡；"
        "安东·尤丁采夫在处置过程中离世；尼基塔·布亚诺夫突发疾病不治。"
    )
    llm37 = FakeLLM(article37)
    ctx37 = build_ctx(llm37)
    plugin37 = make_plugin(module, ctx37)
    ok37, reason37, _ = await plugin37.handle_news_command(**{
        "text": "/新闻三要素",
        "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq",
        "matched_groups": {"material": ""},
        "message": {
            "raw_message": [
                {"type": "reply", "data": {"target_message_content": "昨晚机房空调坏了服务器宕机"}},
                {"type": "text", "data": "/新闻三要素"},
            ],
            "processed_plain_text": "/新闻三要素",
        },
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    })
    assert ok37, f"引用+命令应成功：{reason37}"
    assert sent_all(ctx37), "引用+命令没有发出消息"
    # 假 LLM 返回固定文本，所以要验的是「引用到的素材真的进了提示词」
    assert llm37.prompts, "没有调用模型"
    assert "机房空调坏了" in llm37.prompts[0], (
        f"引用素材没有进入提示词：{llm37.prompts[0][:200]!r}"
    )
    print("[37] 引用 + 命令 完整链路 OK，长度 =", len(sent_body(ctx37)))


    # ── 38. 引用消息时命令前缀能匹配（Reply(...) / [回复了...]） ──
    # 这是真实根因：宿主把引用段拼成 Reply(...) 放在 processed_plain_text 最前面，
    # 严格的 ^/ 锚定会让命令永远匹配不上，表现就是「引用后发指令毫无反应」。
    rp_cases = [
        ("Reply(target_message_id='1', target_sender='小王', target_content='昨晚空调坏了') /新闻三要素", ""),
        ("Reply(target_message_id='1', target_sender='a', target_content='x') /新闻三要素 素材", "素材"),
        ("Reply(target_message_id='1', target_sender='a', target_content='x') @麦麦 /新闻三要素", ""),
        ("[回复了小王的旧消息: 昨晚空调坏了] /新闻三要素", ""),
    ]
    for raw, want_material in rp_cases:
        mm = re.match(pattern, raw)
        assert mm, f"引用前缀导致命令匹配失败：{raw[:60]}"
        assert mm.group("material") == want_material, f"素材解析错误：{raw[:50]} -> {mm.group('material')!r}"
    assert not re.match(pattern, "随便聊聊 /新闻三要素"), "普通聊天不应命中"
    print("[38] 引用前缀下的命令匹配 OK（Reply / [回复了…] / @ 前缀）")

    # ── 39. 从引用前缀里提取被引用正文 ──
    quote_expected = [
        (
            "Reply(target_message_id='1', target_sender='小王', "
            "target_content='昨晚空调坏了') /新闻三要素",
            "昨晚空调坏了",
        ),
        ("[回复了小王的旧消息: 昨晚空调坏了] /新闻三要素", "昨晚空调坏了"),
        ("/新闻三要素 直接素材", ""),
    ]
    for raw, want in quote_expected:
        got = plugin._quoted_text_from_command(raw)
        assert got == want, f"引用正文提取错误：{raw[:50]} -> {got!r}（期望 {want!r}）"
    print("[39] 引用正文提取 OK")

    # ── 40. 引用 + 命令 端到端：素材进提示词、正文发出 ──
    llm40 = FakeLLM(
        "某地发生事故，救援力量已到场处置，周边群众疏散至安全区域，现场秩序正逐步恢复。"
        "值得注意的是，据通报，维亚切斯拉夫·巴兰尼科夫在一线抢险中身亡；"
        "安东·尤丁采夫在处置过程中离世；尼基塔·布亚诺夫突发疾病不治。"
    )
    ctx40 = build_ctx(llm40)
    plugin40 = make_plugin(module, ctx40)
    quoted_cmd = (
        "Reply(target_message_id='1', target_sender='小王', "
        "target_content='昨晚机房空调坏了服务器宕机') /新闻三要素"
    )
    ok40, reason40, _ = await plugin40.handle_news_command(**{
        "text": quoted_cmd,
        "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq",
        "matched_groups": {"material": ""},
        "message": {},
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    })
    assert ok40, f"引用+命令应成功：{reason40}"
    assert sent_all(ctx40), "引用+命令没有发出消息"
    assert llm40.prompts, "没有调用模型"
    assert "机房空调坏了" in llm40.prompts[0], (
        f"引用正文没有进入提示词：{llm40.prompts[0][:200]!r}"
    )
    print("[40] 引用 + 命令 端到端 OK，长度 =", len(sent_body(ctx40)))


    # ── 41. 默认 text_first：先直接发文本，被拦才退回合并转发 ──
    assert plugin.config.output.delivery_mode == "text_first", (
        f"delivery_mode 默认应为 text_first，实际 {plugin.config.output.delivery_mode!r}"
    )
    art41 = (
        "某地发生事故，救援力量已到场处置，周边群众疏散至安全区域，现场秩序正逐步恢复。"
        "值得注意的是，据通报，维亚切斯拉夫·巴兰尼科夫在一线抢险中身亡；"
        "安东·尤丁采夫在处置过程中离世；尼基塔·布亚诺夫突发疾病不治。"
    )
    args41 = {
        "text": "/新闻三要素 某地事故",
        "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq",
        "matched_groups": {"material": "某地事故"},
        "message": {},
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    }

    # 41a：正常情况下直接发文本，不发合并转发（群里不是聊天记录卡片）
    ctx41 = build_ctx(FakeLLM(art41))
    plugin41 = make_plugin(module, ctx41)
    await plugin41.handle_news_command(**args41)
    assert ctx41.send.texts, "默认应直接发文本"
    assert not ctx41.send.forwards, f"文本发送成功时不该再发合并转发：{len(ctx41.send.forwards)}"
    print("[41] 默认直接发文本 OK（不用聊天记录卡片）")

    # 41b：文本发送被拦时，自动退回合并转发
    ctx41b = build_ctx(FakeLLM(art41), flag=True)   # FakeSend 全部失败
    plugin41b = make_plugin(module, ctx41b)
    ok41b, reason41b, _ = await plugin41b.handle_news_command(**args41)
    assert not ok41b, "两种方式都失败时命令应返回失败"
    print("[41b] 两种方式都失败时如实报错 OK")

    # 41c：only-text / only-forward 模式
    cfg41c = plugin.get_default_config()
    cfg41c["output"]["delivery_mode"] = "forward"
    ctx41c = build_ctx(FakeLLM(art41))
    plugin41c = make_plugin(module, ctx41c)
    plugin41c.set_plugin_config(cfg41c)
    await plugin41c.handle_news_command(**args41)
    assert ctx41c.send.forwards and not ctx41c.send.texts, "forward 模式应只发合并转发"

    cfg41d = plugin.get_default_config()
    cfg41d["output"]["delivery_mode"] = "text"
    ctx41d = build_ctx(FakeLLM(art41))
    plugin41d = make_plugin(module, ctx41d)
    plugin41d.set_plugin_config(cfg41d)
    await plugin41d.handle_news_command(**args41)
    assert ctx41d.send.texts and not ctx41d.send.forwards, "text 模式应只发文本"
    print("[41c] 单模式（text / forward）OK")


    # ── 43. 入站 Hook：引用命令归位（让频率插件也认得出命令） ──
    # 引用时宿主把 Reply(...) 拼在文本最前面，导致：
    #   1) 宿主命令匹配（^/ 锚定）失败；
    #   2) 频率插件认不出命令，不建立命令豁免，纯文本被当闲聊拦掉 -> 只能退回聊天记录。
    # Hook 把前缀挪到命令之后，两边都能识别。
    async def hook_case(text):
        msg = {"processed_plain_text": text}
        raw = await plugin.hook_promote_quoted_command(message=msg)
        return runner_apply_hook(raw, {"message": msg}) if raw is not None else None

    hooked = await hook_case(
        "Reply(target_message_id='1', target_sender='小王', target_content='昨晚空调坏了') /新闻三要素"
    )
    assert hooked is not None, "引用命令应被归位"
    assert hooked["message"]["processed_plain_text"] != (
        "Reply(target_message_id='1', target_sender='小王', target_content='昨晚空调坏了') /新闻三要素"
    ), "Hook 返回值必须能被宿主真正应用（modified_kwargs）"
    new_text = hooked["message"]["processed_plain_text"]
    assert new_text.lstrip().startswith("/"), f"归位后应以 / 开头：{new_text[:40]!r}"
    assert "新闻三要素" in new_text
    assert re.match(pattern, new_text), f"归位后宿主命令正则应命中：{new_text[:50]!r}"
    # 归位后素材里不该残留 Reply(...)
    got_quote = plugin._quoted_text_from_command(new_text)
    assert got_quote == "昨晚空调坏了", f"归位后应仍能取到引用正文：{got_quote!r}"
    stripped = plugin._strip_quote_prefix(new_text)
    assert "Reply(" not in stripped, f"素材里的 Reply 前缀应被剥掉：{stripped!r}"
    print("[43] 引用命令入站归位 OK（命令回到开头，引用正文仍可取）")

    # 不该动的消息
    for untouched in [
        "Reply(target_message_id='1', target_content='x') 今天天气不错",
        "Reply(target_message_id='1', target_content='x') /签到",
        "/新闻三要素 美国大选",
        "",
    ]:
        assert await hook_case(untouched) is None, f"不该改写：{untouched[:40]!r}"
    print("[44] Hook 不误伤其它消息 OK")

    # ── 45. 归位后完整跑一遍：直发文本、不产生聊天记录 ──
    art45 = (
        "某地发生事故，救援力量已到场处置，周边群众疏散至安全区域，现场秩序正逐步恢复。"
        "值得注意的是，据通报，维亚切斯拉夫·巴兰尼科夫在一线抢险中身亡；"
        "安东·尤丁采夫在处置过程中离世；尼基塔·布亚诺夫突发疾病不治。"
    )
    llm45 = FakeLLM(art45)
    ctx45 = build_ctx(llm45)
    plugin45 = make_plugin(module, ctx45)
    _msg45 = {
        "processed_plain_text": (
            "Reply(target_message_id='1', target_sender='小王', "
            "target_content='昨晚机房空调坏了服务器宕机') /新闻三要素"
        )
    }
    promoted = runner_apply_hook(
        await plugin45.hook_promote_quoted_command(message=_msg45),
        {"message": _msg45},
    )["message"]["processed_plain_text"]
    ok45, _, _ = await plugin45.handle_news_command(**{
        "text": promoted,
        "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq",
        "matched_groups": {"material": plugin45._material_from_command_text(promoted)},
        "message": {},
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    })
    assert ok45
    assert ctx45.send.texts, "归位后应直接发文本"
    assert not ctx45.send.forwards, "归位后不应再退回聊天记录"
    assert llm45.prompts and "机房空调坏了" in llm45.prompts[0], "引用正文应进入提示词"
    print("[45] 引用命令归位后端到端直发 OK，长度 =", len(ctx45.send.texts[-1]))


    # ── 46. before_process Hook：把 reply 段挪后，让 ^/ 锚定能命中 ──
    # 根因：processed_plain_text 由 process() 按组件顺序拼接；
    # 频率插件的入站判定与本插件同槽位(early)却按 plugin_id 排在前，
    # 所以在 after_process 改文本已经晚了。必须赶在 process() 之前调组件顺序。
    def simulate_process(raw):
        parts = []
        for it in raw:
            t = str(it.get("type") or "").lower()
            if t == "reply":
                data = it.get("data") or {}
                parts.append(
                    f"Reply(target_message_id={data.get('target_message_id')!r}, "
                    f"target_content={data.get('target_content')!r})"
                )
            elif t == "text":
                parts.append(str(it.get("data") or ""))
        return " ".join(p for p in parts if p)

    reorder_cases = [
        [
            {"type": "reply", "data": {"target_message_id": "1", "target_content": "昨晚空调坏了"}},
            {"type": "text", "data": "/新闻三要素"},
        ],
        [
            {"type": "reply", "data": {"target_message_id": "1", "target_content": "昨晚空调坏了"}},
            {"type": "text", "data": "/新闻三要素 美国大选"},
        ],
    ]
    for raw in reorder_cases:
        before = simulate_process(raw)
        # 注意：插件自己的正则已放宽（能容 Reply 前缀），这里不拿它当判据；
        # 真正要保证的是「调整后以 / 开头」——那是频率插件能识别的条件。
        assert not before.lstrip().startswith("/"), f"调整前不应以 / 开头：{before[:50]!r}"
        res = await plugin.hook_reorder_quote_segment(message={"raw_message": list(raw)})
        assert res is not None, f"引用命令应被调整：{before[:50]!r}"
        applied = runner_apply_hook(res, {"message": {"raw_message": list(raw)}})
        applied_raw = applied["message"]["raw_message"]
        assert applied_raw != list(raw), "Hook 返回值必须能被宿主真正应用（modified_kwargs）"
        after = simulate_process(applied_raw)
        assert after.lstrip().startswith("/"), f"调整后应以 / 开头：{after[:50]!r}"
        assert re.match(pattern, after), f"调整后宿主命令正则应命中：{after[:50]!r}"
        assert plugin._quoted_text_from_command(after) == "昨晚空调坏了", (
            f"调整后引用正文应仍可取：{after[:60]!r}"
        )
    print("[46] before_process 调整组件顺序 OK（^/ 能命中，引用正文仍可取）")

    # 不该动的
    for raw in (
        [
            {"type": "reply", "data": {"target_message_id": "1", "target_content": "x"}},
            {"type": "text", "data": "今天天气不错"},
        ],
        [
            {"type": "reply", "data": {"target_message_id": "1", "target_content": "x"}},
            {"type": "text", "data": "/签到"},
        ],
        [{"type": "text", "data": "/新闻三要素 美国大选"}],
        [],
    ):
        assert await plugin.hook_reorder_quote_segment(message={"raw_message": list(raw)}) is None, (
            f"不该调整：{raw[:1]}"
        )
    print("[47] 组件调整 Hook 不误伤其它消息 OK")


    # ── 48. 引用 + 只给名单（不用打素材） ──
    # 用法：引用一条消息，发 /新闻三要素 （张三，李四）
    # 期望：名单替换成功，素材从引用里取。
    def split48(raw):
        return plugin._split_material_and_names(raw)

    mat0, names0 = split48("（张三，李四，王五）")
    assert mat0 == "", f"只给名单时素材应为空：{mat0!r}"
    assert names0 == ["张三", "李四", "王五"], f"名单解析错误：{names0}"

    mat1, names1 = split48("美国大选 （特朗普，奥巴马）")
    assert mat1 == "美国大选" and names1 == ["特朗普", "奥巴马"], f"{mat1!r} / {names1}"

    mat2, names2 = split48("美国大选")
    assert mat2 == "美国大选" and names2 == [], f"{mat2!r} / {names2}"

    mat3, names3 = split48("（）")
    assert mat3 == "（）" and names3 == [], f"空括号应整句当素材：{mat3!r} / {names3}"
    print("[48] 括号名单解析 OK（只给名单 / 素材+名单 / 仅素材 / 空括号）")

    # 端到端：引用 + 只给名单
    art48 = (
        "某地发生事故，救援力量已到场处置，周边群众疏散至安全区域，现场秩序正逐步恢复。"
        "值得注意的是，据通报，张三在一线抢险中身亡；李四在处置过程中离世；王五突发疾病不治。"
    )
    llm48 = FakeLLM(art48)
    ctx48 = build_ctx(llm48)
    plugin48 = make_plugin(module, ctx48)
    _msg48 = {
        "raw_message": [
            {"type": "reply", "data": {"target_message_id": "1", "target_content": "昨晚机房空调坏了服务器宕机"}},
        ],
    }
    ok48, reason48, _ = await plugin48.handle_news_command(**{
        "text": "/新闻三要素 （张三，李四，王五）",
        "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq",
        "matched_groups": {"material": ""},
        "message": _msg48,
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    })
    assert ok48, f"引用+只给名单应成功：{reason48}"
    assert sent_all(ctx48), "应发出正文"
    assert llm48.prompts, "应调用模型"
    prompt48 = llm48.prompts[0]
    # 名单进了提示词
    assert "张三" in prompt48 and "李四" in prompt48 and "王五" in prompt48, "名单没进提示词"
    assert "死亡名单共 **3 人**" in prompt48, "人数应为 3"
    # 默认三人不应出现
    assert "巴兰尼科夫" not in prompt48, "不应再出现默认三人"
    # 素材来自引用
    assert "机房空调坏了" in prompt48, "引用素材没进提示词"
    print("[49] 引用 + 只给名单 端到端 OK（名单替换 + 素材取自引用）")

    # 回归：引用但既无素材也无可读内容时，仍给出提示而不是硬编
    ctx50 = build_ctx(FakeLLM("x"), None)
    plugin50 = make_plugin(module, ctx50)
    ok50, _, _ = await plugin50.handle_news_command(**{
        "text": "/新闻三要素",
        "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq",
        "matched_groups": {"material": ""},
        "message": {"reply_to": "m1"},
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    })
    assert ok50 and sent_all(ctx50), "应给出提示"
    assert "引用" in sent_body(ctx50), f"应提示引用无内容：{sent_body(ctx50)[:40]!r}"
    print("[50] 引用无内容且未给名单时仍提示 OK")


    # ── 51. 新闻社命令：按用户设置，只对自己生效 ──
    STREAM_A = "c58628a91bab2b87117e274bd85f05fa"
    STREAM_B = "1a8eaba25d74b54702ecc69805c231e7"
    UID_A, UID_B = "10001", "20002"

    art51 = (
        "某地发生事故，救援已展开，周边群众疏散至安全区域，现场秩序正逐步恢复。"
        "值得注意的是，据通报，维亚切斯拉夫·巴兰尼科夫在一线抢险中身亡；"
        "安东·尤丁采夫在处置过程中离世；尼基塔·布亚诺夫突发疾病不治。"
    )
    cfg51 = plugin.get_default_config()
    cfg51["output"]["lead_in"] = True
    ctx51 = build_ctx(FakeLLM(art51))
    plugin51 = make_plugin(module, ctx51)
    plugin51.set_plugin_config(cfg51)

    # 设置 A 的新闻社
    ok51, _, _ = await plugin51.handle_agency_command(
        stream_id=STREAM_A, text="/开头署名 观察者报", platform="qq",
        user_id=UID_A, matched_groups={"value": "观察者报"},
    )
    assert ok51, "设置新闻社应成功"
    assert plugin51._agency_for_user(UID_A) == "观察者报"
    assert plugin51._agency_for_user(UID_B) == "盖金社", "别的用户不应受影响"

    # 落盘检查
    saved = plugin51._load_user_profiles()
    assert saved.get(UID_A, {}).get("name") == "观察者报", f"落盘不正确：{saved}"
    assert UID_B not in saved, f"未设置的用户不该落盘：{saved}"

    # A 出稿用观察者报
    await plugin51.handle_news_command(**{
        "text": "/新闻三要素 美国大选", "stream_id": STREAM_A, "platform": "qq",
        "matched_groups": {"material": "美国大选"}, "message": {}, "user_id": UID_A,
        "is_local_operator": False, "plugin_config": {},
    })
    body_a = sent_body(ctx51)
    assert "观察者报" in body_a, f"A 应该用观察者报：{body_a[:30]!r}"
    assert "盖金社" not in body_a, f"A 不该出现盖金社：{body_a[:30]!r}"

    # B 出稿仍用默认
    await plugin51.handle_news_command(**{
        "text": "/新闻三要素 美国大选", "stream_id": STREAM_B, "platform": "qq",
        "matched_groups": {"material": "美国大选"}, "message": {}, "user_id": UID_B,
        "is_local_operator": False, "plugin_config": {},
    })
    body_b = sent_body(ctx51)
    assert "盖金社" in body_b and "观察者报" not in body_b, f"B 应该用默认：{body_b[:30]!r}"
    print("[51] 新闻社按用户隔离 OK（A=观察者报 / B=盖金社）")

    # 重置
    await plugin51.handle_agency_command(
        stream_id=STREAM_A, text="/开头署名 重置", platform="qq",
        user_id=UID_A, matched_groups={"value": "重置"},
    )
    assert plugin51._agency_for_user(UID_A) == "盖金社", "重置后应回到默认"
    assert UID_A not in plugin51._load_user_profiles(), (
        f"重置后不该残留：{plugin51._load_user_profiles()}"
    )
    print("[52] 新闻社重置 OK（回到全局默认并清除记录）")

    # 长度限制
    ok53, _, _ = await plugin51.handle_agency_command(
        stream_id=STREAM_A, text="/开头署名 " + "长" * 30, platform="qq",
        user_id=UID_A, matched_groups={"value": "长" * 30},
    )
    assert not ok53, "超长名称应被拒绝"
    assert "太长" in sent_body(ctx51), f"应提示太长：{sent_body(ctx51)[:30]!r}"
    print("[53] 新闻社名称长度限制 OK")


    # ── 54. 开头署名开关：按用户生效 ──
    # 全局开 -> 用户可单独关；全局关 -> 用户可单独开。
    S_A = "c58628a91bab2b87117e274bd85f05fa"
    S_B = "1a8eaba25d74b54702ecc69805c231e7"
    S_C = "aa11bb22cc33dd44ee55ff66"
    U_A, U_B, U_C = "10001", "20002", "30003"

    def has_lead(text: str) -> bool:
        return bool(re.match(r"^据[^，]{1,12}报道，", text))

    art54 = (
        "某地发生事故，救援已展开，周边群众疏散至安全区域，现场秩序正逐步恢复。"
        "值得注意的是，据通报，维亚切斯拉夫·巴兰尼科夫在一线抢险中身亡；"
        "安东·尤丁采夫在处置过程中离世；尼基塔·布亚诺夫突发疾病不治。"
    )

    # 场景一：全局开，A 单独关
    cfg54 = plugin.get_default_config()
    cfg54["output"]["lead_in"] = True
    ctx54 = build_ctx(FakeLLM(art54))
    plugin54 = make_plugin(module, ctx54)
    plugin54.set_plugin_config(cfg54)

    ok54, _, _ = await plugin54.handle_agency_command(
        stream_id=S_A, text="/开头署名 关", platform="qq", user_id=U_A,
        matched_groups={"value": "关"},
    )
    assert ok54
    assert plugin54._lead_in_for_user(U_A) is False, "A 应被关掉"
    assert plugin54._lead_in_for_user(U_B) is True, "B 应跟随全局（开）"

    await plugin54.handle_news_command(**{
        "text": "/新闻三要素 美国大选", "stream_id": S_A, "platform": "qq",
        "matched_groups": {"material": "美国大选"}, "message": {}, "user_id": U_A,
        "is_local_operator": False, "plugin_config": {},
    })
    assert not has_lead(sent_body(ctx54)), f"A 不该有署名开头：{sent_body(ctx54)[:26]!r}"

    await plugin54.handle_news_command(**{
        "text": "/新闻三要素 美国大选", "stream_id": S_B, "platform": "qq",
        "matched_groups": {"material": "美国大选"}, "message": {}, "user_id": U_B,
        "is_local_operator": False, "plugin_config": {},
    })
    assert has_lead(sent_body(ctx54)), f"B 应该有署名开头：{sent_body(ctx54)[:26]!r}"
    print("[54] 全局开 + 用户单独关 OK（A 无署名 / B 有署名）")

    # 场景二：全局关，A 单独开
    cfg55 = plugin.get_default_config()
    cfg55["output"]["lead_in"] = False
    ctx55 = build_ctx(FakeLLM(art54))
    plugin55 = make_plugin(module, ctx55)
    plugin55.set_plugin_config(cfg55)

    await plugin55.handle_agency_command(
        stream_id=S_A, text="/开头署名 开", platform="qq", user_id=U_A,
        matched_groups={"value": "开"},
    )
    assert plugin55._lead_in_for_user(U_A) is True
    assert plugin55._lead_in_for_user(U_C) is False, "C 应跟随全局（关）"

    await plugin55.handle_news_command(**{
        "text": "/新闻三要素 美国大选", "stream_id": S_A, "platform": "qq",
        "matched_groups": {"material": "美国大选"}, "message": {}, "user_id": U_A,
        "is_local_operator": False, "plugin_config": {},
    })
    assert has_lead(sent_body(ctx55)), f"A 应该有署名：{sent_body(ctx55)[:26]!r}"

    await plugin55.handle_news_command(**{
        "text": "/新闻三要素 美国大选", "stream_id": S_C, "platform": "qq",
        "matched_groups": {"material": "美国大选"}, "message": {}, "user_id": U_C,
        "is_local_operator": False, "plugin_config": {},
    })
    assert not has_lead(sent_body(ctx55)), f"C 不该有署名：{sent_body(ctx55)[:26]!r}"
    print("[55] 全局关 + 用户单独开 OK（A 有署名 / C 无署名）")

    # 旧格式（字符串）兼容
    legacy = plugin55._agency_file()
    assert legacy is not None
    legacy.write_text(json.dumps({"99999": "老格式报"}, ensure_ascii=False), encoding="utf-8")
    profiles56 = plugin55._load_user_profiles()
    assert profiles56.get("99999", {}).get("name") == "老格式报", f"旧格式兼容失败：{profiles56}"
    assert plugin55._agency_for_user("99999") == "老格式报"
    print("[56] 旧版存储格式（纯字符串）兼容 OK")


    # ── 57. 工具与 API 入口也要遵守启停/禁用流（审核建议） ──
    art57 = (
        "某地发生事故，救援已展开，周边群众疏散至安全区域，现场秩序正逐步恢复。"
        "值得注意的是，据通报，维亚切斯拉夫·巴兰尼科夫在一线抢险中身亡；"
        "安东·尤丁采夫在处置过程中离世；尼基塔·布亚诺夫突发疾病不治。"
    )
    SID57 = "c58628a91bab2b87117e274bd85f05fa"

    # 停用后，工具不应出稿
    cfg57 = plugin.get_default_config()
    cfg57["plugin"]["enabled"] = False
    ctx57 = build_ctx(FakeLLM(art57))
    plugin57 = make_plugin(module, ctx57)
    plugin57.set_plugin_config(cfg57)
    plugin57._last_stream_id = SID57
    r57 = await plugin57.handle_tool_write_news(material="某地事故")
    assert not r57["success"], f"停用后工具不该成功：{r57}"
    assert "停用" in r57["content"], f"应说明插件已停用：{r57['content'][:40]!r}"
    assert not sent_all(ctx57), "停用后不应发送任何消息"

    r57b = await plugin57.api_generate_news("某地事故")
    assert not r57b["success"], f"停用后 API 也不该出稿：{r57b}"
    print("[57] 插件停用后工具与 API 均不出稿 OK")

    # 禁用的聊天流，工具也不应出稿
    cfg58 = plugin.get_default_config()
    cfg58["plugin"]["blocked_stream_ids"] = [SID57]
    ctx58 = build_ctx(FakeLLM(art57))
    plugin58 = make_plugin(module, ctx58)
    plugin58.set_plugin_config(cfg58)
    plugin58._last_stream_id = SID57
    r58 = await plugin58.handle_tool_write_news(material="某地事故")
    assert not r58["success"], f"禁用流不该成功：{r58}"
    assert "禁用" in r58["content"], f"应说明聊天流已禁用：{r58['content'][:40]!r}"
    assert not sent_all(ctx58), "禁用流不应发送"
    print("[58] 禁用的聊天流下工具不出稿 OK")

    # 启用状态 + 正常流，工具应正常出稿（确认守卫没误伤）
    cfg59 = plugin.get_default_config()
    ctx59 = build_ctx(FakeLLM(art57))
    plugin59 = make_plugin(module, ctx59)
    plugin59.set_plugin_config(cfg59)
    plugin59._last_stream_id = SID57
    r59 = await plugin59.handle_tool_write_news(material="某地事故")
    assert r59["success"], f"正常情况应成功：{r59}"
    assert sent_all(ctx59), "正常情况应发送"
    print("[59] 启用状态下工具正常出稿 OK（守卫未误伤）")


    # ── 60. 平台限制：三个入口行为一致（第二次审核点出的唯一不一致） ──
    art60 = (
        "某地发生事故，救援已展开，周边群众疏散至安全区域，现场秩序正逐步恢复。"
        "值得注意的是，据通报，维亚切斯拉夫·巴兰尼科夫在一线抢险中身亡；"
        "安东·尤丁采夫在处置过程中离世；尼基塔·布亚诺夫突发疾病不治。"
    )
    SID60 = "c58628a91bab2b87117e274bd85f05fa"

    cfg60 = plugin.get_default_config()
    cfg60["plugin"]["only_platforms"] = ["qq"]
    ctx60 = build_ctx(FakeLLM(art60))
    plugin60 = make_plugin(module, ctx60)
    plugin60.set_plugin_config(cfg60)

    # 命令入口：qq 放行
    ok60, _, _ = await plugin60.handle_news_command(**{
        "text": "/新闻三要素 某地事故", "stream_id": SID60, "platform": "qq",
        "matched_groups": {"material": "某地事故"}, "message": {}, "user_id": "u",
        "is_local_operator": False, "plugin_config": {},
    })
    assert ok60, "qq 平台应放行"
    # 命令入口应记下「会话 -> 平台」
    assert plugin60._stream_platform.get(SID60) == "qq", (
        f"应记录会话所属平台：{plugin60._stream_platform}"
    )
    assert plugin60._platform_allowed_for_stream(SID60) is True
    print("[60] 命令入口记录会话平台 OK")

    # 工具入口：同一会话应继承 qq 的放行结果
    # （先清掉冷却记录，否则会被节流挡住，测不到平台判定）
    plugin60._last_call.pop(SID60, None)
    r60 = await plugin60.handle_tool_write_news(stream_id=SID60, material="某地事故")
    assert r60["success"], f"工具入口应随会话继承平台判定：{r60}"
    print("[61] 工具入口继承会话平台判定 OK")

    # 若该会话属于未允许的平台，工具也应拒绝
    cfg62 = plugin.get_default_config()
    cfg62["plugin"]["only_platforms"] = ["qq"]
    ctx62 = build_ctx(FakeLLM(art60))
    plugin62 = make_plugin(module, ctx62)
    plugin62.set_plugin_config(cfg62)
    plugin62._stream_platform["telegram_stream_abc123"] = "telegram"
    r62 = await plugin62.handle_tool_write_news(
        stream_id="telegram_stream_abc123", material="某地事故"
    )
    assert not r62["success"], f"未允许的平台应被拒绝：{r62}"
    assert "平台" in r62["content"], f"应说明平台未启用：{r62['content'][:40]!r}"
    assert not sent_all(ctx62), "未允许平台不应发送"
    print("[62] 工具入口拒绝未启用平台 OK")

    # API 同理
    r63 = await plugin62.api_generate_news("某地事故", stream_id="telegram_stream_abc123")
    assert not r63["success"], f"API 也应拒绝未启用平台：{r63}"
    # 没传 stream_id 时无从判断，放行
    r64 = await plugin62.api_generate_news("某地事故")
    assert r64["success"], f"未传 stream_id 时应放行：{r64}"
    print("[63] API 入口平台限制与放行 OK")

    # 未配置平台限制时，三个入口都放行
    cfg65 = plugin.get_default_config()
    ctx65 = build_ctx(FakeLLM(art60))
    plugin65 = make_plugin(module, ctx65)
    plugin65.set_plugin_config(cfg65)
    plugin65._last_stream_id = SID60
    assert plugin65._platform_allowed_for_stream(SID60) is True, "未配置限制应放行"
    r65 = await plugin65.handle_tool_write_news(stream_id=SID60, material="某地事故")
    assert r65["success"]
    print("[64] 未配置平台限制时全放行 OK")


    # ── 65. 图片素材识别与展开 ──
    # 宿主把图片转成 "[图片：xxx]" 塞进文本；引用图片当素材时要先展开成可写的现场描述，
    # 否则模型会把 "[图片：…]" 这层壳原样写进新闻。
    img_cases = [
        ("[图片：一只橘猫趴在键盘上]", True),
        ("[【图片：会议室里人们围着白板】]", True),
        ("[表情包：熊猫头大笑]", True),
        ("[image: a cat on a keyboard]", True),
    ]
    for raw, want_img in img_cases:
        out, is_img = plugin._normalize_material(raw)
        assert is_img is want_img, f"图片识别错误：{raw!r} -> {is_img}"
        # 素材只保留画面描述本身，任何指令性文字都不能混进来——
        # 否则模型会把指令原样抄进正文。
        assert "[图片" not in out and "[表情包" not in out, f"展开后不应残留标记：{out!r}"
        assert "请把画面" not in out and "不要出现" not in out, (
            f"素材里不该混入写作指令：{out!r}"
        )
        assert out and len(out) > 2, f"展开后应留有画面描述：{out!r}"

    # 普通方括号素材不能被误判
    for plain in ["[新闻] 某地发生事故", "【公告】服务器维护", "美国大选", "（张三，李四）"]:
        out, is_img = plugin._normalize_material(plain)
        assert not is_img, f"不该误判为图片：{plain!r}"
        assert out == plain, f"普通素材不应被改写：{plain!r} -> {out!r}"

    # 图文混排：就地展开
    mixed = "看到这一幕 [图片：一只橘猫趴在键盘上] 大家都笑了"
    out_mixed, is_mixed = plugin._normalize_material(mixed)
    assert is_mixed and "[图片" not in out_mixed, f"混排展开失败：{out_mixed!r}"
    assert "一只橘猫趴在键盘上" in out_mixed, f"混排应保留画面描述：{out_mixed!r}"
    assert "看到这一幕" in out_mixed, f"混排应保留原有文字：{out_mixed!r}"
    print("[65] 图片素材识别与展开 OK")

    # ── 66. 引用图片 + 命令：端到端出稿 ──
    art66 = (
        "某地一家公司的办公室里，一只橘猫跳上工位趴在键盘上，导致正在处理的文件被误改，"
        "同事发现后将其抱离，工作随即恢复。值得注意的是，据通报，"
        "维亚切斯拉夫·巴兰尼科夫在驱赶时被猫抓伤后感染身亡；"
        "安东·尤丁采夫在搬运主机时触电离世；尼基塔·布亚诺夫突发疾病不治。"
    )
    llm66 = FakeLLM(art66)
    ctx66 = build_ctx(llm66)
    plugin66 = make_plugin(module, ctx66)
    _msg66 = {
        "raw_message": [
            {"type": "reply", "data": {
                "target_message_id": "1",
                "target_message_content": "[图片：一只橘猫趴在键盘上，旁边有半杯咖啡]",
            }},
        ],
    }
    ok66, reason66, _ = await plugin66.handle_news_command(**{
        "text": "/新闻三要素",
        "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq",
        "matched_groups": {"material": ""},
        "message": _msg66,
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    })
    assert ok66, f"引用图片+命令应成功：{reason66}"
    assert sent_all(ctx66), "应发出正文"
    assert llm66.prompts, "应调用模型"
    prompt66 = llm66.prompts[0]
    # 画面内容进了提示词，且没有残留 [图片：…] 的壳
    assert "橘猫趴在键盘上" in prompt66, "图片描述没进提示词"
    assert "[图片：" not in prompt66, f"不该把图片标记原样喂给模型：{prompt66[:200]!r}"
    # 图片场景的指令应在提示词的指令区出现，且不能出现在素材块里
    assert "这条素材是对一张照片的画面描述" in prompt66, "应给出图片场景指令"
    material_block = prompt66.split("【素材】")[1].split("【硬性要求】")[0]
    assert "这条素材是对一张照片的画面描述" not in material_block, (
        f"指令不该混进素材块：{material_block[:120]!r}"
    )
    print("[66] 引用图片 + 命令 端到端 OK")

    # ── 67. 引用图片 + 名单：两者可同时用 ──
    llm67 = FakeLLM(
        "某地办公室发生意外，一只橘猫跳上工位导致文件被误改，随后被同事抱离，"
        "工作秩序恢复正常。值得注意的是，据通报，张三在处置中身亡；"
        "李四在搬运设备时遇难。"
    )
    ctx67 = build_ctx(llm67)
    plugin67 = make_plugin(module, ctx67)
    ok67, reason67, _ = await plugin67.handle_news_command(**{
        "text": "/新闻三要素 （张三，李四）",
        "stream_id": "1a8eaba25d74b54702ecc69805c231e7",
        "platform": "qq",
        "matched_groups": {"material": ""},
        "message": _msg66,
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    })
    assert ok67, f"引用图片+名单应成功：{reason67}"
    p67 = llm67.prompts[0]
    assert "橘猫趴在键盘上" in p67, "图片描述应进提示词"
    assert "张三" in p67 and "李四" in p67, "名单应生效"
    print("[67] 引用图片 + 自定义名单 OK")

    # ── 68. 括号内含方括号时，引用提取不截断 ──
    got68 = plugin._quoted_text_from_command(
        "[回复了小王的图片: [图片：一只橘猫趴在键盘上]]"
    )
    assert got68 == "[图片：一只橘猫趴在键盘上]", f"嵌套方括号提取错误：{got68!r}"
    got68b = plugin._quoted_text_from_command("[回复了小王的旧消息: 昨晚空调坏了]")
    assert got68b == "昨晚空调坏了", f"普通引用提取错误：{got68b!r}"
    print("[68] 嵌套方括号的引用提取 OK")


    # ── 69. 空壳素材拦截（宿主只给了 "[emoji]" 时不该硬编） ──
    # 真实事故：引用一张表情包发命令，宿主把它识别成 EMOJI 且 vlm_processed=0，
    # processed_plain_text 只有 "[emoji]"，插件拿不到任何画面信息，模型只能凭空编。
    shell_cases = ["[emoji]", "[表情包]", "[image]", "[图片]", "[动图]", "emoji", "［表情包］"]
    for raw in shell_cases:
        assert plugin._is_empty_shell(raw), f"应识别为空壳：{raw!r}"

    # 有实际内容的都不算空壳
    content_cases = [
        "[图片：一只橘猫趴在键盘上]",
        "[表情包: 呆萌,可爱,卖萌]",
        "美国大选",
        "昨晚空调坏了",
    ]
    for raw in content_cases:
        assert not plugin._is_empty_shell(raw), f"不该判为空壳：{raw!r}"

    # 端到端：空壳素材应给出提示，而不是硬编一篇
    llm69 = FakeLLM("这段不该被用上")
    ctx69 = build_ctx(llm69)
    plugin69 = make_plugin(module, ctx69)
    _msg69 = {
        "raw_message": [
            {"type": "reply", "data": {"target_message_id": "1", "target_message_content": "[emoji]"}},
        ],
    }
    ok69, _, _ = await plugin69.handle_news_command(**{
        "text": "/新闻三要素",
        "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq",
        "matched_groups": {"material": ""},
        "message": _msg69,
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    })
    assert ok69
    assert sent_all(ctx69), "应给出提示"
    tip = sent_body(ctx69)
    assert "瞎编" in tip, f"应说明硬写会瞎编：{tip[:60]!r}"
    assert "表情包" in tip, f"表情包应单独说明：{tip[:80]!r}"
    assert not llm69.prompts, f"空壳素材不该调用模型硬编：{len(llm69.prompts)} 次"
    print("[69] 空壳素材拦截 OK（不调用模型，直接说明原因）")

    # 但「空壳 + 自定义名单」仍允许：用户可能只想要名单效果
    llm70 = FakeLLM(
        "某地发生事故，救援已展开，周边群众疏散至安全区域，现场秩序正逐步恢复。"
        "值得注意的是，据通报，张三在一线抢险中身亡；李四在处置过程中离世。"
    )
    ctx70 = build_ctx(llm70)
    plugin70 = make_plugin(module, ctx70)
    ok70, _, _ = await plugin70.handle_news_command(**{
        "text": "/新闻三要素 （张三，李四）",
        "stream_id": "1a8eaba25d74b54702ecc69805c231e7",
        "platform": "qq",
        "matched_groups": {"material": ""},
        "message": {"raw_message": [{"type": "reply", "data": {"target_message_content": "[emoji]"}}]},
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    })
    assert ok70
    assert llm70.prompts, "带名单时应照常生成"
    print("[70] 空壳 + 自定义名单仍可出稿 OK")


    # ── 71. 空壳素材从 images 表恢复描述 ──
    # 真实事故：宿主来不及在入库前跑完图片识别，processed_plain_text 是空的，
    # 引用时只剩 "[image]"，但描述其实已经在 images 表里（vlm_processed=1）。
    class _FakeDB:
        def __init__(self, mapping): self.mapping = mapping
        async def query(self, model_name, query_type="get", filters=None, limit=None,
                        single_result=False):
            # 宿主 database.query 直接返回结果本身（single_result=True 时是 dict 或 None），
            # 没有 {"success":..., "data":...} 包装。
            h = (filters or {}).get("image_hash", "")
            desc = self.mapping.get(h, "")
            return {"description": desc, "image_type": "IMAGE"} if desc else None

    class _FakeMsgWithHash:
        def __init__(self, mapping): self.mapping = mapping
        async def get_by_id(self, mid, chat_id="", **kw):
            # 复刻宿主真实结构：
            #   1) 外面包一层 {"success":..., "message": {...}}
            #   2) hash 在消息段的**顶层**，不在 data 里
            return {
                "success": True,
                "message": {
                    "message_id": mid,
                    "raw_message": [{"type": "image", "data": "", "hash": "abc123def456"}],
                    "processed_plain_text": "",
                },
            }

    mapping71 = {"abc123def456": "一只橘猫趴在键盘上，桌上有半杯咖啡"}
    ctx71 = build_ctx(FakeLLM("x"))
    ctx71.db = _FakeDB(mapping71)
    ctx71.message = _FakeMsgWithHash(mapping71)
    plugin71 = make_plugin(module, ctx71)

    msg71 = {"raw_message": [
        {"type": "reply", "data": {"target_message_id": "m1", "target_message_content": "[image]"}},
    ]}
    recovered = await plugin71._recover_image_material(msg71, "s")
    assert recovered == "[图片：一只橘猫趴在键盘上，桌上有半杯咖啡]", f"恢复失败：{recovered!r}"
    print("[71] 从 images 表恢复图片描述 OK")

    # 查不到描述时返回空（不编造）
    ctx72 = build_ctx(FakeLLM("x"))
    ctx72.db = _FakeDB({})
    ctx72.message = _FakeMsgWithHash({})
    plugin72 = make_plugin(module, ctx72)
    assert await plugin72._recover_image_material(msg71, "s") == "", "查不到时应返回空"
    print("[72] 表里没有描述时不编造 OK")

    # 端到端：空壳 + 表里有描述 -> 正常出稿
    art73 = (
        "某地一家游戏公司的开发间里，屏幕上正在调试一款格斗小游戏，"
        "两个黄色圆润角色对峙，界面显示血量与计时。值得注意的是，据通报，"
        "维亚切斯拉夫·巴兰尼科夫在调试中因设备故障身亡；"
        "安东·尤丁采夫在排查线路时遇难；尼基塔·布亚诺夫突发疾病不治。"
    )
    llm73 = FakeLLM(art73)
    ctx73 = build_ctx(llm73)
    ctx73.db = _FakeDB(mapping71)
    ctx73.message = _FakeMsgWithHash(mapping71)
    plugin73 = make_plugin(module, ctx73)
    ok73, reason73, _ = await plugin73.handle_news_command(**{
        "text": "/新闻三要素",
        "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq",
        "matched_groups": {"material": ""},
        "message": msg71,
        "user_id": "u",
        "is_local_operator": False,
        "plugin_config": {},
    })
    assert ok73, f"应成功：{reason73}"
    assert llm73.prompts, "应从表里补到描述并调用模型"
    assert "橘猫趴在键盘上" in llm73.prompts[0], "补回的描述应进提示词"
    print("[73] 空壳恢复后端到端出稿 OK")


    # ── 74. 引用图片 + 文字素材：两者必须结合（不能只取文字） ──
    # 真实问题：命令后写了素材时，图片会被文字盖掉。
    IMG_HASH = "abc123def456"
    mapping74 = {IMG_HASH: "画面里两个黄色角色在屏幕上对峙，配有血条和计时"}

    def _mk74():
        class _DB:
            async def query(self, model_name, query_type="get", filters=None, limit=None,
                            single_result=False):
                h = (filters or {}).get("image_hash", "")
                d = mapping74.get(h, "")
                return {"description": d, "image_type": "IMAGE"} if d else None

        class _Msg:
            async def get_by_id(self, mid, chat_id="", **kw):
                return {"success": True, "message": {
                    "raw_message": [{"type": "image", "data": "", "hash": IMG_HASH}],
                    "processed_plain_text": "",
                }}

        llm = FakeLLM(
            "某项目团队本周开展版本联调，准备上线。值得注意的是，据通报，"
            "甲一在连续联调后突发疾病身亡；乙二在赶赴现场途中遇难；丙三在调试设备时意外离世。"
        )
        c74 = build_ctx(llm)
        c74.db = _DB()
        c74.message = _Msg()
        p74 = make_plugin(module, c74)
        return p74, llm, c74

    # 只给名单 + 引用图片：素材应来自图片
    p74a, llm74a, _ = _mk74()
    await p74a.handle_news_command(**{
        "text": "/新闻三要素 （甲一，乙二，丙三）",
        "stream_id": "c58628a91bab2b87117e274bd85f05fa",
        "platform": "qq",
        "matched_groups": {"material": "（甲一，乙二，丙三）"},
        "message": {"raw_message": [{"type": "reply", "data": {"target_message_id": "1"}}]},
        "user_id": "u", "is_local_operator": False, "plugin_config": {},
    })
    pr74a = llm74a.prompts[0]
    assert "黄色角色" in pr74a, "只给名单时应采用图片内容"
    assert "甲一" in pr74a and "乙二" in pr74a, "名单应生效"
    print("[74] 引用图片 + 只给名单 OK（素材来自图片，名单生效）")

    # 文字素材 + 名单 + 引用图片：三者都要在提示词里
    p74b, llm74b, _ = _mk74()
    cmd74 = "某项目团队本周开展版本联调，准备上线"
    await p74b.handle_news_command(**{
        "text": f"/新闻三要素 {cmd74} （甲一，乙二，丙三）",
        "stream_id": "1a8eaba25d74b54702ecc69805c231e7",
        "platform": "qq",
        "matched_groups": {"material": f"{cmd74} （甲一，乙二，丙三）"},
        "message": {"raw_message": [{"type": "reply", "data": {"target_message_id": "1"}}]},
        "user_id": "u", "is_local_operator": False, "plugin_config": {},
    })
    pr74b = llm74b.prompts[0]
    assert cmd74 in pr74b, f"文字素材应保留：{pr74b[:300]!r}"
    assert "黄色角色" in pr74b, f"图片内容也应保留（两者结合）：{pr74b[:300]!r}"
    assert "甲一" in pr74b and "丙三" in pr74b, "名单应生效"
    material_block74 = pr74b.split("【素材】")[1].split("【硬性要求】")[0]
    assert "结合起来" in material_block74, f"应明确要求结合：{material_block74[:200]!r}"
    if "。。" in pr74b:
        raise AssertionError("合并时不该出现重复句号")
    print("[75] 引用图片 + 文字素材 + 名单 OK（三者结合，无重复标点）")

    # ── 76. 引用图片时的名单替换（专项确认） ──
    p76, llm76, _ = _mk76 = _mk74()
    await p76.handle_news_command(**{
        "text": "/新闻三要素 （张三，李四）",
        "stream_id": "aa11bb22cc33dd44ee55ff66",
        "platform": "qq",
        "matched_groups": {"material": "（张三，李四）"},
        "message": {"raw_message": [{"type": "reply", "data": {"target_message_id": "1"}}]},
        "user_id": "u", "is_local_operator": False, "plugin_config": {},
    })
    pr76 = llm76.prompts[0]
    assert "死亡名单共 **2 人**" in pr76, f"应为 2 人：{pr76[:400]!r}"
    assert "张三" in pr76 and "李四" in pr76, "名单应替换"
    assert "巴兰尼科夫" not in pr76, "默认三人不该出现"
    print("[76] 引用图片时名单替换 OK（2 人，默认三人不出现）")

    print("\n全部自检通过 ✅")
    return 0























if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
