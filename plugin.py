"""新闻三要素 —— 按用户素材写一篇新闻体短文。

这是一个纯玩梗插件：用户发来一段素材（事件、公告、吐槽、游戏更新说明都行），
插件会以新闻通稿的笔调写一篇短消息，并把「维亚切斯拉夫·巴兰尼科夫」
「安东·尤丁采夫」「尼基塔·布亚诺夫」三人的死讯自然地织进这篇报道里。
正文中的死亡设定全部是虚构的，输出默认带一行声明。

命令：``/新闻三要素 <素材>``（唯一入口）。

能力使用：
- ``llm.generate``     生成正文（失败时回退到本地模板，保证命令永远有输出）
- ``send.text``        发送正文
- ``send.forward``     正文过长时改用合并转发
- ``message.get_by_id`` 读取被引用消息作为素材
"""

import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from maibot_sdk import API, Command, Field, HomeCard, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import (
    ErrorPolicy,
    HookMode,
    HookOrder,
    ToolParameterInfo,
    ToolParamType,
)

# 命令正则：只有 /新闻三要素 一个入口，后面可跟任意素材。
# Host 用 re.search 匹配，因此要自带 ^ 锚点。
#
# 开头这一串前缀是踩坑换来的，必须在命令前允许出现：
#   1. ``Reply(...)``：**引用消息时**宿主把引用段转成
#      ``Reply(target_message_id=..., target_sender=..., target_content=...)``
#      并拼在 processed_plain_text 最前面。不加这条，引用后发命令永远匹配不上，
#      表现就是「引用消息再发指令毫无反应」。
#   2. ``[回复了××的消息: ...]``：老格式的引用前缀。
#   3. ``@机器人``：QQ 客户端在引用时常自动插入 @。
# 容器用 (?P<quoted>...) 捕获，用来从里面把引用正文抠出来当素材。
COMMAND_PATTERN = (
    r"^(?:\s*"
    r"(?P<quote_struct>Reply\([^()]*\))"
    r"|\s*(?P<quote_brackets>\[回复了[^\]]*\]|\[引用[^\]]*\])"
    r"|\s*@[^\s　/／]{1,40}\s*"
    r")*"
    r"\s*[/／]新闻三要素[\s　]*(?P<material>[\s\S]*)$"
)
COMMAND_TIMEOUT_MS = 120_000

# 标题/署名命令：/开头署名 <名称> / /开头署名 开|关 / /开头署名 重置
AGENCY_PATTERN = r"^[/／]开头署名(?:[\s　]+(?P<value>[\s\S]*))?$"

_HELP_TOKENS = {"帮助", "help", "?", "？", "说明", "用法", "-h", "--help"}
_RETRY_TOKENS = {"重写", "重来", "再来", "再来一篇", "再写一篇"}

# 单个名字的最大长度；括号里超长的片段会被丢弃，避免把整句塞进名单。
_NAME_MAX_LEN = 20


# ─────────────────────────── 配置模型 ───────────────────────────


class PluginSection(PluginConfigBase):
    """插件基础设置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "newspaper"
    __ui_order__ = 0

    config_version: str = Field(
        default="1.0.0",
        description="配置版本号",
        json_schema_extra={
            "label": "配置版本",
            "hint": "升级插件时用来判断是否要重建配置。一般不用动，改了会导致配置按新结构重建"
        },
    )
    enabled: bool = Field(
        default=True,
        description="插件总开关",
        json_schema_extra={
            "label": "总开关","hint": "关掉后 /新闻三要素 不再响应，消息按普通聊天处理"},
    )
    cooldown_seconds: int = Field(
        default=20,
        description="冷却时间（秒）",
        json_schema_extra={
            "label": "冷却时间（秒）",
            "hint": "同一聊天流两次出稿的间隔。0 表示不限制；正文发送失败时不会记冷却，可立即重试"
        },
    )
    only_platforms: List[str] = Field(
        default_factory=list,
        description="限定生效的平台",
        json_schema_extra={
            "label": "限定平台","hint": "平台名一般是 qq / telegram / xiaoheihe，留空即全部放行"},
    )
    blocked_stream_ids: List[str] = Field(
        default_factory=list,
        description="禁用的聊天流",
        json_schema_extra={
            "label": "禁用的聊天流","hint": "填聊天流 ID，每行一个。命中则静默不响应，适合给特定群关掉"},
    )


class GenerationConfig(PluginConfigBase):
    """生成设置。"""

    __ui_label__ = "生成"
    __ui_icon__ = "sparkles"
    __ui_order__ = 1

    task_name: str = Field(
        default="",
        description="模型任务名",
        json_schema_extra={
            "label": "模型任务名",
            "hint": "填宿主模型配置里的任务名，如 utils / replyer / planner。留空则用默认任务（utils）"
        },
    )
    model_name: str = Field(
        default="",
        description="直选具体模型名",
        json_schema_extra={
            "label": "具体模型名",
            "hint": "填 [[models]] 里的 name 可绕过任务策略直达该模型。"
            "与任务名同时填写时，任务名决定参数、模型名决定用哪个模型；不需要就留空"
        },
    )
    temperature: float = Field(
        default=0.9,
        description="采样温度",
        json_schema_extra={
            "label": "采样温度","hint": "0~2。越高越发散、越容易写出花样；新闻体建议 0.7~1.0，想更稳可降到 0.6"},
    )
    max_tokens: int = Field(
        default=4096,
        description="单次生成上限（token，含思考链）",
        json_schema_extra={
            "label": "生成上限（token）",
            "hint": "开启思考的模型会先花大量 token 做推理。给太小会被思考吃光，正文写不出来，"
            "部分 provider 还会把思考原文当正文返回。建议 ≥4096",
        },
    )
    target_length: int = Field(
        default=220,
        description="期望正文长度（字）",
        json_schema_extra={
            "label": "期望长度（字）",
            "hint": "只是给模型的建议值，实际受 max_chars 硬截断。名单人多时建议一起调大，否则后半截会挤"
        },
    )
    pivot: str = Field(
        default="值得注意的是",
        description="转折词",
        json_schema_extra={
            "label": "转折词",
            "hint": "正文前半截写素材延伸，到这个词开始写死讯。整篇仍是一段话，"
            "转折词前必须是句号或分号。可换成「另据通报」「与此同时」等"
        },
    )
    max_attempts: int = Field(
        default=3,
        description="不合格重试次数",
        json_schema_extra={
            "label": "重试次数",
            "hint": "正文被判定不合格（缺人 / 像思考链 / 分段了 / 结构不对）时最多重写几次。"
            "次数越多越慢，全部失败会退回本地模板。低于 1 按 1 处理"
        },
    )
    extra_style: str = Field(
        default="",
        description="附加写作要求",
        json_schema_extra={
            "label": "附加写作要求",
            "hint": "会追加到提示词末尾，用来微调文风。留空则不加",
            "placeholder": "例如：多用短句；避免网络流行语；保持通稿的克制口吻",
        },
    )


class PersonConfig(PluginConfigBase):
    """一位当事人的署名信息。

    身份（``title``）刻意留空：默认由模型按新闻场景现编——写战争就是军官/士兵，
    写事故就是工人/司机，写体坛就是教练/运动员。只有填了才锁定。
    """

    name: str = Field(
        default="",
        description="姓名",
        json_schema_extra={
            "hint": "写全会写进正文。中文用「张」「王五」这类；外文名建议带间隔号，如「安东尼·布朗」",
            "placeholder": "例如：张三 / 维亚切斯拉夫·巴兰尼科夫",
        },
    )
    title: str = Field(
        default="",
        description="固定身份（可选）",
        json_schema_extra={
            "hint": "留空 = 由模型按新闻场景现编身份（战争就是军官，事故就是车间主管，体育就是教练）。"
            "填了就锁死这个身份，不再随场景变化",
            "placeholder": "例如：某部指挥官 / 车间主管",
        },
    )
    cause_hint: str = Field(
        default="",
        description="固定死因（可选）",
        json_schema_extra={
            "hint": "只在走本地回退模板时使用；留空则由模板按场景自动生成。"
            "正常走模型时，死因一律由模型按素材现场编",
            "placeholder": "例如：在阵地转移途中遭炮击身亡",
        },
    )


class PeopleConfig(PluginConfigBase):
    """当事人名单。

    默认是固定的三人；想换人、加人、减人都改 ``list``：
    ``["张三", "李四", "王五", "赵六"]`` 或带身份的 ``[{name="…", title="…"}]``。
    """

    __ui_label__ = "当事人"
    __ui_icon__ = "users"
    __ui_order__ = 2

    list: List[PersonConfig] = Field(
        default_factory=lambda: [
            PersonConfig(name="维亚切斯拉夫·巴兰尼科夫"),
            PersonConfig(name="安东·尤丁采夫"),
            PersonConfig(name="尼基塔·布亚诺夫"),
        ],
        description="默认死亡名单",
        json_schema_extra={
            "label": "默认死亡名单",
            "hint": "不写人名时就用这份名单。每项至少要填 name；填了 title 就锁定该人身份。"
            "想临时换人不用改这里，直接在命令后用括号写名字：/新闻三要素 素材（张三，李四）"
        },
    )
    max_people: int = Field(
        default=8,
        description="名单人数上限",
        json_schema_extra={
            "label": "人数上限",
            "hint": "名单和命令里超出这个数的名字会被忽略。人多时后半截很长，"
            "建议同时调大 target_length 和 max_chars"
        },
    )


class OutputConfig(PluginConfigBase):
    """输出设置。"""

    __ui_label__ = "输出"
    __ui_icon__ = "send"
    __ui_order__ = 3

    set_reply: bool = Field(
        default=False,
        description="是否引用用户消息（⚠️ 保持关闭）",
        json_schema_extra={
            "label": "引用用户消息（勿开）",
            "hint": "⚠️ 保持 false。插件层拿不到被回复的消息对象，"
            "一旦开启，宿主会直接丢弃整条消息，表现是「后台有日志但群里没反应」",
        },
    )
    headline_prefix: str = Field(
        default="",
        description="标题前缀",
        json_schema_extra={
            "label": "标题前缀",
            "hint": "加在正文最前面的固定文字。留空不加。"
            "想标明是玩梗可以填「【虚构】」",
            "placeholder": "例如：【虚构】",
        },
    )
    disclaimer: bool = Field(
        default=False,
        description="是否追加虚构声明",
        json_schema_extra={
            "label": "追加虚构声明",
            "hint": "开启后会在正文末尾另起一段加「本文为虚构创作…」，与哀悼句分开。默认关闭"
        },
    )
    disclaimer_text: str = Field(
        default="（本文为虚构创作 · 纯属玩梗 · 与现实中任何人物、机构及事件无关）",
        description="虚构声明内容",
        json_schema_extra={
            "label": "声明内容",
            "hint": "仅在「追加虚构声明」开启时生效，可自由改写",
            "placeholder": "（本文为虚构创作 · 纯属玩梗 · 与现实无关）",
        },
    )
    max_chars: int = Field(
        default=400,
        description="正文硬上限（字）",
        json_schema_extra={
            "label": "正文硬上限（字）",
            "hint": "超出会按句子边界截断，且会保住结尾的哀悼句。名单人数多时要调大"
        },
    )
    forward_threshold: int = Field(
        default=500,
        description="改用合并转发的字数阈值",
        json_schema_extra={
            "label": "合并转发阈值",
            "hint": "正文超过这个字数就改用合并转发（否则单条太长）。0 表示永远发单条文本"
        },
    )
    delivery_mode: Literal["text_first", "forward_first", "text", "forward"] = Field(
        default="text_first",
        description="正文发送方式",
        json_schema_extra={
            "label": "发送方式",
            "hint": "text_first（推荐）：直接发文本，被出站拦截时才退回合并转发；"
            "forward_first：先合并转发（稳定过拦截，但群里显示为聊天记录卡片）；"
            "text：只发文本；forward：只发合并转发",
        },
    )
    forward_nickname: str = Field(
        default="盖金社",
        description="合并转发里的署名",
        json_schema_extra={
            "label": "转发署名","hint": "走合并转发时，气泡里显示的发件人名称"},
    )
    lead_in: bool = Field(
        default=False,
        description="开头署名（默认关闭）",
        json_schema_extra={
            "label": "开头署名",
            "hint": "开启后在正文最前面加「据○○社报道，」。"
            "模型自己写了电头会自动去重，不会出现两句。默认关闭。",
        },
    )
    agency: str = Field(
        default="盖金社",
        description="署名用的机构名",
        json_schema_extra={
            "label": "署名机构名",
            "hint": "即「据___报道」里的名字，仅在「开头署名声」开启时生效",
            "placeholder": "例如：盖金社 / 路透社",
        },
    )


class EndingConfig(PluginConfigBase):
    """结尾句。

    结尾句里用 ``{verb}`` 表示那个可替换的词；插件每次从 ``verbs`` 里随机挑一个填进去。
    """

    __ui_label__ = "结尾"
    __ui_icon__ = "heart"
    __ui_order__ = 5

    enabled: bool = Field(
        default=True,
        description="启用统一结尾句",
        json_schema_extra={
            "label": "总开关",
            "hint": "开启后插件会剥掉模型自写的「仍在调查中」之类套话，换成下面的结尾句。关掉则保留模型原结尾"
        },
    )
    template: str = Field(
        default="让我们为他们{verb}🙏",
        description="结尾句模板",
        json_schema_extra={
            "label": "结尾模板",
            "hint": "用 {verb} 表示那个随机词的位置。例如「让我们为他们{verb}🙏」「愿他们{verb}」",
            "placeholder": "例如：让我们为他们{verb}🙏",
        },
    )
    verbs: List[str] = Field(
        default_factory=lambda: ["作哀", "悼观", "设敬", "应敬", "应悼"],
        description="候选词列表",
        json_schema_extra={
            "label": "候选词",
            "hint": "每次出稿从这里随机挑一个填进 {verb}。只留一个词就是固定用它；留空则回落到「作哀」",
            "placeholder": "例如：作哀, 悼观, 设敬",
        },
    )
    use_emoji: bool = Field(
        default=True,
        description="自动补 🙏",
        json_schema_extra={
            "label": "自动补表情",
            "hint": "模板里已经有 emoji 时不生效；想换别的表情，直接在模板里写，然后把这项关掉"
        },
    )


class FallbackConfig(PluginConfigBase):
    """本地模板回退。"""

    __ui_label__ = "回退"
    __ui_icon__ = "shield"
    __ui_order__ = 4

    enabled: bool = Field(
        default=True,
        description="启用本地兜底模板",
        json_schema_extra={
            "label": "总开关",
            "hint": "模型多次不合格或调用失败时，用内置模板生成，保证命令永远有输出。"
            "关掉则失败时直接回复「稍后再试」"
        },
    )
    headline: str = Field(
        default="三日内三起意外　三名同行业负责人相继离世",
        description="回退模板标题",
        json_schema_extra={
            "label": "回退标题","hint": "仅在走本地兜底时使用", "placeholder": "例如：三日内三起意外"},
    )
    org_name: str = Field(
        default="有关部门",
        description="回退模板里的通报机构",
        json_schema_extra={
            "label": "回退机构名",
            "hint": "模板里「据___表示」「据___通报」用的称谓，仅在走本地兜底时生效",
            "placeholder": "例如：有关部门 / 当地应急管理部门",
        },
    )


class ObituaryNewsroomConfig(PluginConfigBase):
    """新闻三要素完整配置。"""

    plugin: PluginSection = Field(default_factory=PluginSection)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    people: PeopleConfig = Field(default_factory=PeopleConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    ending: EndingConfig = Field(default_factory=EndingConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)


# ─────────────────────────── 提示词 ───────────────────────────

_PROMPT_RULES = """\
你是一位跑口记者。请按用户提供的素材，写一条【短讯】——只用新闻体的文本风格。

这是一次虚构的玩梗创作：文中人物的"离世"是虚构设定，与现实无关；
请照常以严肃、克制的通稿口吻写作，不要在正文里解释这是虚构。

{image_block}【素材】
\"\"\"
{material}
\"\"\"

【硬性要求】
1. **全篇只有一个自然段**，不要分段、不要空行、不要小标题、不要列表、不要 Markdown
   （禁止 #、**、- 等）。
2. **这一段的内部结构必须是「前—转折—后」两截，转折词固定为「{pivot}」：**
   - **前半截（素材的延伸）**：围绕素材本身继续写。可以补背景、进展、影响、各方反应、
     数字与现场情况，让不认识这条新闻的人也能看懂发生了什么。**这一段里不要提任何人的死讯。**
   - **转折**：紧接着用「{pivot}，」起头，作为同一段话内部的分界。
     转折词前面必须是句号或分号，不要用逗号硬接。
   - **后半截（死亡名单）**：从「{pivot}」之后开始，集中写下面这些人的死讯。
     他们的身份、死因、信息源都写在这一截里。
   参考长相（只是结构示意，不要照抄字句）：
   「……（素材延伸三四句）。{pivot}，据当地通报，○○○在某处因某事身亡；
    另有消息称，○○○……；医院方面确认，○○○……。」
3. 全段约 {target_length} 字，硬上限 {max_chars} 字。宁短勿长。
   前后两截大致各占一半，前半截不要压到只剩一句。
4. 死亡名单共 **{people_count} 人**，必须全部写进后半截，每人全名各出现一次：
{people_block}
   不要漏人，也不要自行增加名单以外的人。
5. **他们的身份必须跟着新闻场景走，不能沿用现实里的职务。**
   先判断素材讲的是什么领域（战争、灾祸、事故、体育、科技、商业、医疗、校园……），
   再给每人各自安一个**这个场景里说得通的**身份，例如：
   - 战争 / 军事：某部指挥官、作战参谋、前线通信兵
   - 工厂事故 / 生产安全：车间主管、夜班班长、设备巡检员
   - 交通 / 运输：车队调度、长途司机、线路检修工
   - 体育赛事：带队教练、队医、随队翻译
   - 科研 / 医疗：课题组负责人、主治医师、设备工程师
   - 商业 / 金融：门店经理、风控专员、区域负责人
   各人身份要**互不相同**，也不要都用同一种死法。
{lock_block}6. 死因必须从素材的场景里长出来，而且要**各不相同**、听上去像真的会发生：
   战场上死于炮击或误伤、事故里倒塌或中毒、连夜抢修后突发疾病、赶赴现场途中车祸……
   不写猎奇死因，不写血污细节，用词克制、庄重。
7. 融入要自然：不要写成"讣告名单"、排比句或并列堆砌。
   用「当地通报」「军方声明」「两名不愿具名的同事」「医院方面」这类信息源，把各人的死讯
   串成同一条新闻里的事实，而不是一份份独立讣告。
8. 素材里的关键事实、数字、地名、时间尽量保留；素材没有的不要编造可核查的具体细节
   （精确门牌号、证件号、真实公司全称），需要时用「当地」「有关部门」等模糊表达。
9. **不要在结尾写"相关情况仍在调查中""仍在进一步核实"这类套话**，
   结尾由系统统一补上。写到最后一个死讯即可收住。
10. 直接输出这一段成品正文：不要写分析、规划、提纲、自问自答，
    不要复述要求，不要写"标题："、"正文："、"字数："之类的标签，不要加任何声明或注释。
{extra_style}"""


# 素材本身就写了死亡事件时使用：不另起转折，直接把死者换成名单里的人。
_PROMPT_REPLACE = """\
你是一位跑口记者。请按用户提供的素材，写一条【短讯】——只用新闻体的文本风格。

这是一次虚构的玩梗创作：文中人物的"离世"是虚构设定，与现实无关；
请照常以严肃、克制的通稿口吻写作，不要在正文里解释这是虚构。

{image_block}【素材】
\"\"\"
{material}
\"\"\"

【重要背景】
这条素材里**已经写了有人死亡**。不要另外虚构一起新的死亡事件，
也不要用「{pivot}」这类转折句再补一段死讯。
正确做法是：**把素材里的死者直接替换成下面名单里的人**，其余事件经过按素材写。

【硬性要求】
1. **全篇只有一个自然段**，不要分段、不要空行、不要小标题、不要列表、不要 Markdown
   （禁止 #、**、- 等）。
2. **不要使用「{pivot}」这类转折结构。** 整段顺着素材原本的叙事写下去，
   该交代的背景、经过、影响、各方反应照常写，死亡就发生在素材原本写死亡的那个位置。
3. 全段约 {target_length} 字，硬上限 {max_chars} 字。宁短勿长。
4. 把素材里的死者换成这 **{people_count} 人**，每人全名各出现一次：
{people_block}
   不要漏人，也不要自行增加名单以外的人。
   - 素材写"几人死亡"就按名单人数写；素材只说"有人死亡"也要把名单全部写进去。
   - **保留素材原本的死因与场景**（火灾就是火灾，空难就是空难），
     不要把素材里的事故改写成别的原因。
5. **他们的身份按素材场景设定**，每人各自一个、互不相同，例如：
   - 战争 / 军事：某部指挥官、作战参谋、前线通信兵
   - 事故 / 生产安全：车间主管、夜班班长、设备巡检员
   - 交通 / 运输：车队调度、长途司机、线路检修工
   - 体育赛事：带队教练、队医、随队翻译
   - 科研 / 医疗：课题组负责人、主治医师、设备工程师
   若素材已经写明他们的身份，直接沿用，不要另编。
{lock_block}6. 死因沿用素材里的那起事件；如果素材只写了"死亡"没写具体死因，
   就按素材场景给出一句克制、可信的交代。不写猎奇细节，不写血污描写。
7. 素材里的关键事实、数字、地名、时间尽量保留；素材没有的不要编造可核查的具体细节
   （精确门牌号、证件号、真实公司全称），需要时用「当地」「有关部门」等模糊表达。
8. **不要在结尾写"相关情况仍在调查中""仍在进一步核实"这类套话**，
   结尾由系统统一补上。
9. 直接输出这一段成品正文：不要写分析、规划、提纲、自问自答，
   不要复述要求，不要写"标题："、"正文："、"字数："之类的标签，不要加任何声明或注释。
{extra_style}"""


# 判断素材里是否已经包含死亡事件。
_DEATH_WORDS = (
    "死亡", "身亡", "遇难", "丧生", "罹难", "遇害", "殉职", "牺牲",
    "去世", "离世", "逝世", "病逝", "不治", "罹", "毙命", "殒命",
    "dead", "died", "killed", "death", "casualties", "fatalities",
)


def _material_has_death(material: str) -> bool:
    """判断素材里是否已经写了死亡事件。

    命中则改用「替换死者」模式：不再另起转折虚构一起死亡，
    而是把素材里的死者换成配置名单里的人。
    """

    text = str(material or "").lower()
    return any(word in text for word in _DEATH_WORDS)


def _active_people(people: PeopleConfig, roster: Optional[List[PersonConfig]] = None) -> list[PersonConfig]:
    """返回当前生效的死亡名单（按顺序，去掉空名与超出上限的项）。

    Args:
        people: 配置里的默认名单。
        roster: 本次调用临时指定的名单；为空时用配置里的默认名单。

    Returns:
        list[PersonConfig]: 实际要写进新闻的名单。
    """

    source = list(roster) if roster else list(getattr(people, "list", []) or [])
    limit = max(1, int(getattr(people, "max_people", 8) or 8))
    result: list[PersonConfig] = []
    for person in source:
        if not isinstance(person, PersonConfig):
            continue
        if not str(person.name or "").strip():
            continue
        result.append(person)
        if len(result) >= limit:
            break
    return result


def _format_people_block(people: PeopleConfig, roster: Optional[List[PersonConfig]] = None) -> str:
    """把死亡名单排成提示词里的条目；身份留空时让模型自己定。"""

    lines: list[str] = []
    for person in _active_people(people, roster):
        name = str(person.name or "").strip()
        title = str(person.title or "").strip()
        lines.append(f"   - {name}" if not title else f"   - {name}（身份固定为：{title}）")
    return "\n".join(lines) if lines else "   - （未配置当事人）"


def _format_lock_block(people: PeopleConfig, roster: Optional[List[PersonConfig]] = None) -> str:
    """全部当事人身份都被配置锁定时，提醒模型别再自编身份。"""

    active = _active_people(people, roster)
    fixed = [str(person.title or "").strip() for person in active]
    if active and all(fixed):
        return f"   （本插件已固定上面 {len(active)} 人的身份，请直接使用，不要另编。）\n"
    return ""


# ─────────────────────────── 插件主体 ───────────────────────────


class ObituaryNewsroom(MaiBotPlugin):
    """把素材加工成新闻体短文，并把三人的死讯织进去。"""

    config_model = ObituaryNewsroomConfig

    def __init__(self) -> None:
        super().__init__()
        self._last_call: Dict[str, float] = {}
        self._last_material: Dict[str, str] = {}
        self._counter: int = 0
        # 最近一次真实会话 ID：模型调工具时常把 stream_id 编成 "current"，
        # 那时用它兜底。
        self._last_stream_id: str = ""
        # 会话 -> 平台 映射：工具/API 入口拿不到 platform，
        # 只能靠命令入口记下的这一份反查，才能让三个入口的守卫行为一致。
        self._stream_platform: Dict[str, str] = {}
        # 最近一次素材的图片类型："" / "image" / "emoji"。
        # 用来在拼提示词时选择对应的指令段。
        self._last_material_kind: str = ""

    # ── 生命周期 ──

    async def on_load(self) -> None:
        cfg = self.config
        self.ctx.logger.info(
            "盖金社已就位：冷却=%ss，回退模板=%s",
            cfg.plugin.cooldown_seconds,
            cfg.fallback.enabled,
        )

    async def on_unload(self) -> None:
        self._last_call.clear()
        self._last_material.clear()

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        self.ctx.logger.info("盖金社配置更新：scope=%s version=%s", scope, version)

    # ── Hook：让「引用 + 命令」也能被当成命令 ──

    @HookHandler(
        "chat.receive.before_process",
        name="news_quote_reorder",
        description="引用消息发命令时，把 reply 段挪到文本段之后，让命令出现在文本开头",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,
        timeout_ms=2000,
    )
    async def hook_reorder_quote_segment(self, message: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """在 ``message.process()`` **之前**调整组件顺序。

        为什么必须在这个 hook：``processed_plain_text`` 是在
        ``SessionMessage.process()`` 里按组件顺序拼接出来的，
        而「发言频率控制」的入站判定只读这一份文本、并且和本插件同槽位
        （``EARLY``）却按 ``plugin_id`` 排在前（``github.*`` < ``local.*``）。
        也就是说，等 ``chat.receive.after_process`` 才改文本，它已经判完了。

        所以要在 ``process()`` 之前把 ``reply`` 段挪到最后：
        拼出来的文本就会是 ``/新闻三要素 Reply(...)``，
        ``^/`` 一锚就中，它认得出来这是命令，才会为该会话建立 60 秒命令豁免。
        """

        del kwargs
        if not isinstance(message, dict):
            return None

        raw = message.get("raw_message")
        if not isinstance(raw, list) or len(raw) < 2:
            return None

        # 先看看这条消息处理后会是什么样：只有「引用 + 命令」才动手
        reply_items = [
            item for item in raw
            if isinstance(item, dict) and str(item.get("type") or "").lower() == "reply"
        ]
        if not reply_items:
            return None

        text_items = [
            item for item in raw
            if isinstance(item, dict) and str(item.get("type") or "").lower() == "text"
        ]
        if not text_items:
            return None

        # 文本段里第一段可读文字必须以 / 开头且含本插件命令名
        first_text = str(text_items[0].get("data") or "").strip()
        if not first_text.startswith(("/", "／")):
            return None
        if "新闻三要素" not in first_text[:40]:
            return None

        # 把 reply 段整体挪到末尾，其余保持原顺序
        reordered = [item for item in raw if item not in reply_items]
        reordered.extend(reply_items)

        message = dict(message)
        message["raw_message"] = reordered
        self.ctx.logger.info("已把引用段移到文本之后，确保命令出现在开头")
        # Runner 只认 "modified_kwargs"，其它键会被直接忽略（踩过坑）。
        return {"action": "continue", "modified_kwargs": {"message": message}}


    @HookHandler(
        "chat.receive.after_process",
        name="news_quote_promote",
        description="引用消息发命令时，把 Reply(...) 前缀挪到命令之后，避免命令被识别成聊天",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,
        timeout_ms=2000,
    )
    async def hook_promote_quoted_command(self, message: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """把「Reply(...) /新闻三要素」整形成命令本身。

        为什么需要：引用消息时宿主会把引用段转成
        ``Reply(target_message_id=..., target_content=...)`` 并拼在
        ``processed_plain_text`` 最前面。于是：

        - 宿主的命令匹配（``^/`` 锚定）匹配不到；
        - 「发言频率控制」也认不出这是命令，**不会为该会话建立命令豁免**，
          导致插件发出的纯文本被当成闲聊取消掉，只能退回合并转发（显示成聊天记录）。

        这里在入站阶段把这段前缀**挪到命令后面**：命令回到开头，
        两边都能识别；引用正文仍然留在文本里，插件照常能抠出来当素材。
        """

        del kwargs
        if not isinstance(message, dict):
            return None

        text = str(message.get("processed_plain_text") or "")
        if not text:
            return None

        match = self._LEADING_QUOTE_PREFIX.match(text)
        if not match:
            return None
        prefix = match.group("prefix")
        rest = text[match.end():].lstrip()
        if not rest:
            return None
        # 只处理真正的本插件命令，别碰别人的消息。
        if not rest.startswith(("/", "／")):
            return None
        if "新闻三要素" not in rest[:40]:
            return None

        new_text = f"{rest} {prefix}".strip()
        message = dict(message)
        message["processed_plain_text"] = new_text
        self.ctx.logger.info("引用命令已归位，交给后续识别：%s", new_text[:60])
        # Runner 只认 "modified_kwargs"，其它键会被直接忽略（踩过坑）。
        return {"action": "continue", "modified_kwargs": {"message": message}}

    # 开头可能出现的引用前缀（与 COMMAND_PATTERN 保持一致）
    _LEADING_QUOTE_PREFIX = re.compile(
        r"^(?P<prefix>"
        r"\s*Reply\([^()]*\)"
        r"|\s*\[回复了[^\]]*\]"
        r"|\s*\[引用[^\]]*\]"
        r")+",
        re.IGNORECASE,
    )

    # ── 命令 ──

    @Command(
        "news_agency",
        description="设置自己的新闻社名与开头署名开关（只对自己生效）",
        pattern=AGENCY_PATTERN,
        timeout_ms=COMMAND_TIMEOUT_MS,
    )
    async def handle_agency_command(
        self,
        stream_id: str = "",
        text: str = "",
        platform: str = "",
        user_id: str = "",
        matched_groups: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[str], bool]:
        """处理 /开头署名 命令。

        - ``/开头署名 观察者报`` —— 设置自己的新闻社（只影响自己触发的新闻）
        - ``/开头署名``        —— 查看当前设置
        - ``/开头署名 重置``   —— 恢复为默认
        """

        del kwargs

        if not self.config.plugin.enabled:
            return False, "插件已停用", False
        if not self._platform_allowed(platform):
            return False, "当前平台未启用", False

        target = self._sanitize_stream_id(stream_id) or self._last_stream_id or stream_id
        uid = str(user_id or "").strip()
        if not uid:
            await self.ctx.send.text("拿不到你的用户 ID，没法保存专属新闻社。", target)
            return False, "缺少 user_id", True

        groups = matched_groups if isinstance(matched_groups, dict) else {}
        value = str(groups.get("value") or "").strip()
        if not value:
            match = re.match(AGENCY_PATTERN, str(text or "").strip())
            if match:
                value = str(match.group("value") or "").strip()

        profiles = self._load_user_profiles()
        profile = dict(profiles.get(uid) or {})
        default_agency = str(self.config.output.agency or "盖金社").strip() or "盖金社"
        default_lead = bool(self.config.output.lead_in)

        lowered = value.lower()
        wants_show = (not value) or value in {"查看", "?", "？", "show"}
        wants_reset = value in {"重置", "恢复", "默认", "reset"}
        # 开关只认简单的「开 / 关」；别名仅作兜底。
        wants_on = lowered in {"开", "开启", "on"}
        wants_off = lowered in {"关", "关闭", "off"}

        # 查看
        if wants_show:
            current_name = str(profile.get("name") or "").strip() or default_agency
            current_lead = profile.get("lead_in", default_lead)
            name_src = "你自己设置的" if profile.get("name") else "全局默认"
            lead_src = "你自己设置的" if "lead_in" in profile else "全局默认"
            await self.ctx.send.text(
                f"新闻社：「{current_name}」（{name_src}）\n"
                f"开头署名：{'开' if current_lead else '关'}（{lead_src}）\n"
                "· 改名：/开头署名 新名称\n"
                "· 开关开头署名：/开头署名 开  或  /开头署名 关\n"
                "· 全部恢复默认：/开头署名 重置",
                target,
            )
            return True, "已查看新闻社设置", True

        # 重置
        if wants_reset:
            if uid in profiles:
                profiles.pop(uid, None)
                if not self._save_user_profiles(profiles):
                    await self.ctx.send.text("保存失败，检查插件数据目录权限。", target)
                    return False, "保存失败", True
                await self.ctx.send.text(
                    f"已全部恢复为全局设置：新闻社「{default_agency}」，"
                    f"开头署名{'开' if default_lead else '关'}。",
                    target,
                )
            else:
                await self.ctx.send.text(
                    f"你本来就用全局设置：新闻社「{default_agency}」，"
                    f"开头署名{'开' if default_lead else '关'}。",
                    target,
                )
            return True, "已重置新闻社设置", True

        # 开关开头署名
        if wants_on or wants_off:
            profile["lead_in"] = wants_on
            profiles[uid] = profile
            if not self._save_user_profiles(profiles):
                await self.ctx.send.text("保存失败，检查插件数据目录权限。", target)
                return False, "保存失败", True
            current_name = str(profile.get("name") or "").strip() or default_agency
            if wants_on:
                await self.ctx.send.text(
                    f"好，以后你的新闻会以「据{current_name}报道，」开头。（只对你生效）\n"
                    "想关掉：/开头署名 关",
                    target,
                )
            else:
                await self.ctx.send.text(
                    "好，以后你的新闻不再加「据○○社报道，」开头。（只对你生效）\n"
                    "想打开：/开头署名 开",
                    target,
                )
            return True, "已设置开头署名", True

        # 设置名称
        if len(value) > 20:
            await self.ctx.send.text("名字太长了，控制在 20 字以内。", target)
            return False, "名称过长", True
        if re.search(r"[\n\r]", value):
            await self.ctx.send.text("名字里不能有换行。", target)
            return False, "名称非法", True

        profile["name"] = value
        profiles[uid] = profile
        if not self._save_user_profiles(profiles):
            await self.ctx.send.text("保存失败，检查插件数据目录权限。", target)
            return False, "保存失败", True

        self.ctx.logger.info("用户 %s 的新闻社设为「%s」", uid, value)
        lead_now = profile.get("lead_in", default_lead)
        tip = (
            f"以后你发的新闻会以「据{value}报道，」开头。"
            if lead_now
            else "（开头署名当前是关的，想加的话发 /开头署名 开）"
        )
        await self.ctx.send.text(
            f"好，以后你发的新闻都由「{value}」报道。（只对你生效）\n{tip}",
            target,
        )
        return True, "已设置新闻社", True

    @Command(
        "news_three_elements",
        description="按用户提供的素材写新闻体短文；括号内可指定死亡名单",
        pattern=COMMAND_PATTERN,
        timeout_ms=COMMAND_TIMEOUT_MS,
    )
    async def handle_news_command(
        self,
        stream_id: str = "",
        text: str = "",
        platform: str = "",
        matched_groups: Optional[Dict[str, Any]] = None,
        message: Optional[Dict[str, Any]] = None,
        user_id: str = "",
        **kwargs: Any,
    ) -> Tuple[bool, Optional[str], bool]:
        """处理 /新闻三要素 命令。"""

        del kwargs

        # 前置拦截一律「不发言、也不拦消息」：返回的第三项是拦截标志，
        # 置 False 让这条命令继续走正常聊天流程，别把用户的普通消息吃掉。
        if not self.config.plugin.enabled:
            return False, "插件已停用", False
        if not self._platform_allowed(platform):
            return False, "当前平台未启用", False
        if stream_id in set(self.config.plugin.blocked_stream_ids):
            return False, "当前聊天流已禁用", False

        # 命令这条路拿到的 stream_id 是宿主给的真实会话 ID，记下来给工具兜底。
        real_stream = self._sanitize_stream_id(stream_id)
        if real_stream:
            self._last_stream_id = real_stream
            # 顺手记住这个会话属于哪个平台，供工具/API 入口反查。
            normalized_platform = str(platform or "").strip().lower()
            if normalized_platform:
                self._stream_platform[real_stream] = normalized_platform

        groups = matched_groups if isinstance(matched_groups, dict) else {}
        raw_material = str(groups.get("material") or "").strip()
        if not raw_material:
            raw_material = self._material_from_command_text(text)

        # 先剥掉引用前缀再拆名单：否则 Reply(...) 结尾的括号会被误当成名单
        # （真实用户名里也常见「（完全推移）」这种括号）。
        raw_material = self._strip_quote_prefix(raw_material)

        # 再拆「素材 / 括号名单」：引用场景下可能只给了名单，素材要从引用里取。
        material, names = self._split_material_and_names(raw_material)
        roster: Optional[List[PersonConfig]] = None
        if names:
            roster = self._roster_from_names(names)
            self.ctx.logger.info("本次指定死亡名单：%s", "、".join(names))

        # 引用消息时，宿主把引用段拼进了命令文本（Reply(...) / [回复了…]）。
        # 入站 Hook 可能已把它挪到末尾，所以素材要同时清掉首/尾两处残留前缀。
        if material:
            cleaned = self._strip_quote_prefix(material)
            if cleaned != material:
                material = cleaned

        # 诊断：把 entry 阶段拿到的东西记下来，排查「引用图片却写成别的内容」时用。
        log_debug = getattr(self.ctx.logger, "debug", None)
        if callable(log_debug):
            log_debug(
                "命令素材溯源：text=%r raw_segments=%s material=%r names=%s",
                str(text or "")[:200],
                [
                    {
                        "type": s.get("type"),
                        "content": str((s.get("data") or {}).get("target_message_content") or "")[:80]
                        if isinstance(s.get("data"), dict)
                        else "",
                    }
                    for s in ((message or {}).get("raw_message") or [])
                    if isinstance(s, dict)
                ],
                material[:120],
                names,
            )

        # 素材为空时，按可信度取引用正文：
        #   1) message.raw_message 的 reply 段（最接近原始数据）
        #   2) 命令文本里的 Reply(...) 前缀（可能是被截断的预览）
        if not material and message:
            material = await self._material_from_quote(message, stream_id)

        if not material:
            quoted = self._quoted_text_from_command(text)
            if quoted:
                self.ctx.logger.info("从命令文本的引用前缀取到素材，长度=%s", len(quoted))
                material = quoted

        # 这条消息到底有没有引用？没有引用时不该报"引用取不到内容"，
        # 而应该走"裸命令 -> 发帮助说明"的老路径。
        has_quote = self._has_quote(message, text)

        if not material and not names and has_quote:
            # 引用了消息却取不到文字：明确告诉用户，别拿空素材硬编新闻。
            await self.ctx.send.text(
                "没读到你引用那条消息的文字内容（可能是图片、转发或已过期）。"
                "直接把素材打出来吧：/新闻三要素 <素材>",
                self._sanitize_stream_id(stream_id) or stream_id,
            )
            return True, "引用内容无文字", True

        # 素材可能是引用图片得到的「[图片：xxx]」，先展开成可写的现场描述。
        material, from_image = self._normalize_material(material)
        if from_image:
            self.ctx.logger.info(
                "素材来自%s，已展开（长度=%s）",
                "表情包" if self._last_material_kind == "emoji" else "图片",
                len(material),
            )

        # 空壳素材先尝试从 images 表把描述补回来：
        # 宿主有时来不及在入库前跑完识别，正文是空的，但描述已经写进表里了。
        #
        # 注意这里**不能**要求 material 非空：引用场景下 material 常被剥成空串
        # （命令本身没有附素材），那恰恰是最需要去表里捞的时候。
        if message and self._is_empty_shell(material):
            recovered = await self._recover_image_material(message, stream_id)
            if recovered:
                material, from_image = self._normalize_material(recovered)
                self.ctx.logger.info("空壳素材已从图片描述恢复（长度=%s）", len(material))

        # 仍然拿不到画面内容，写出来必然是编的。
        # 与其硬编一篇对不上的新闻，不如说清楚原因。
        if not names and self._has_quote(message, text) and self._is_empty_shell(material):
            self.ctx.logger.warning("素材为空壳标记，拒绝出稿：%r", material[:60])
            await self.ctx.send.text(
                "这张图我没能读取出画面内容（宿主只给了空标记，通常是表情包或识别没跑完），"
                "硬写会变成瞎编。\n"
                "可以试试：换一张图，或直接把素材打出来 —— /新闻三要素 <素材>",
                self._sanitize_stream_id(stream_id) or stream_id,
            )
            return True, "图片无可读内容", True

        # 帮助/重写这类裸命令要能穿过冷却，否则连说明都拿不到。
        if material in _HELP_TOKENS:
            await self.ctx.send.text(self._help_text(), stream_id)
            return True, "已发送使用说明", True

        # 「重写」：沿用上一次同流的素材，方便对草稿不满意时再要一版。
        if material in _RETRY_TOKENS:
            previous = self._last_material.get(stream_id, "")
            if not previous:
                await self.ctx.send.text("这家报纸还没出过稿，先给我点素材吧。", stream_id)
                return True, "无可重写的素材", True
            material = previous

        if not material:
            await self.ctx.send.text(self._help_text(), stream_id)
            return True, "素材为空，已发送使用说明", True

        wait = self._cooldown_remaining(stream_id)
        if wait > 0:
            await self.ctx.send.text(f"新闻社还在排字，{wait} 秒后再试。", stream_id)
            return True, "冷却中", True

        user_agency = self._agency_for_user(user_id)
        user_lead = self._lead_in_for_user(user_id)

        ok, article = await self._generate_news(material, roster, user_agency, user_lead, from_image)
        if not ok:
            await self.ctx.send.text("编辑部暂时联系不上记者，稍后再试。", stream_id)
            return False, "生成失败", True

        deliver_to = self._sanitize_stream_id(stream_id) or self._last_stream_id or stream_id
        delivered = await self._deliver(article, deliver_to, user_agency)
        if not delivered:
            # 发送失败就不要记冷却，否则用户重试会被"还在排字"挡住，看起来像彻底没反应。
            self.ctx.logger.error("正文未能送达，本次不进入冷却（stream=%s）", stream_id)
            return False, "发送失败", True

        self._remember(stream_id, material)
        self.ctx.logger.info(
            "已生成讣告新闻：stream=%s length=%s user=%s", stream_id, len(article), user_id or "?"
        )
        return True, None, True

    # ── 工具：供麦麦在对话里直接调用 ──

    @Tool(
        "write_obituary_news",
        description=(
            "按用户提供的素材写一篇短篇新闻（新闻通稿文体），并把虚构人物的死讯自然织进报道。"
            "默认写配置里的三人；用户用括号点名了别人时，把人名传进 names 覆盖默认名单。"
            "当用户说“帮我写篇新闻/讣告/通稿”，或给了一段素材要求改写成新闻时调用。"
            "素材里没有的内容不要自行补齐；生成后会自动发送到当前聊天。"
        ),
        parameters=[
            ToolParameterInfo(
                name="stream_id",
                param_type=ToolParamType.STRING,
                description="当前聊天流 ID。不要自己编，留空即可——插件会自动用当前会话",
                required=False,
            ),
            ToolParameterInfo(
                name="material",
                param_type=ToolParamType.STRING,
                description="用户提供的素材原文（事件经过、公告、吐槽、更新说明等）",
                required=True,
            ),
            ToolParameterInfo(
                name="names",
                param_type=ToolParamType.ARRAY,
                description="可选：本次要写进新闻的人名列表，留空则用默认三人",
                required=False,
                # 数组参数必须给 items_schema，否则宿主枚举工具时抛
                # ValueError("数组参数必须提供 items_schema")，会把聊天循环一起带崩。
                items_schema={"type": "string", "description": "人名"},
            ),
        ],
        timeout_ms=COMMAND_TIMEOUT_MS,
    )
    async def handle_tool_write_news(
        self,
        stream_id: str = "",
        material: str = "",
        names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """工具入口：生成并发送。"""

        del kwargs

        # 与命令入口保持同一套守卫：插件停用后，模型仍可能调用本工具，
        # 不拦的话会出现「关了插件还在出稿」。
        if not self.config.plugin.enabled:
            return {"success": False, "content": "插件已停用，未生成。"}

        material = str(material or "").strip()
        if not material:
            return {"success": False, "content": "素材为空，无法写作。请先向用户索取素材。"}

        # 模型若把 "[图片：xxx]" 原样当素材传进来，同样展开成画面描述。
        material, from_image = self._normalize_material(material)

        # 模型经常把 stream_id 编成 "current" / "当前会话" 这类占位符，
        # 用它发送会直接失败。这类值一律丢弃，回落到最近一次真实会话。
        stream_id = self._sanitize_stream_id(stream_id)
        if not stream_id:
            stream_id = self._last_stream_id
        if not stream_id:
            return {
                "success": False,
                "content": "拿不到当前会话 ID，没法发送。请让用户直接发命令 /新闻三要素 <素材>。",
            }

        # 禁用的聊天流同样不响应。
        if stream_id in set(self.config.plugin.blocked_stream_ids):
            return {"success": False, "content": "当前聊天流已禁用，未生成。"}

        # 平台限制：工具拿不到 platform，靠命令入口记下的「会话 -> 平台」反查。
        # 查不到（例如该会话从没跑过本插件命令）时按放行处理，
        # 避免因为缺乏信息就拒绝正常请求。
        platform_ok = self._platform_allowed_for_stream(stream_id)
        if platform_ok is False:
            return {"success": False, "content": "当前平台未启用，未生成。"}

        wait = self._cooldown_remaining(stream_id)
        if wait > 0:
            return {
                "success": False,
                "content": f"同一聊天流 {int(self.config.plugin.cooldown_seconds)} 秒内只写一篇，请等待 {wait} 秒。",
            }

        roster = None
        clean_names = [str(n).strip() for n in (names or []) if str(n).strip()]
        if clean_names:
            roster = self._roster_from_names(clean_names)

        ok, article = await self._generate_news(
            material, roster, self._agency_for_user(""), None, from_image
        )
        if not ok:
            return {"success": False, "content": "生成失败，请稍后再试。"}

        if stream_id:
            await self._deliver(article, stream_id)
        self._remember(stream_id, material)
        return {"success": True, "content": article}

    # ── 插件间 API：只出稿，不发送 ──

    @API("generate_obituary_news", description="按素材生成讣告新闻正文，不发送", version="1", public=True)
    async def api_generate_news(
        self,
        material: str,
        stream_id: str = "",
        names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """给其他插件调用的出稿接口。

        只出稿不发送；插件停用时同样返回失败，避免「关了还在后台跑」。
        """

        if not self.config.plugin.enabled:
            return {"success": False, "text": "", "stream_id": stream_id}

        # 传了 stream_id 就按它反查平台；没传则无从判断，直接放行。
        if str(stream_id or "").strip():
            if self._platform_allowed_for_stream(stream_id) is False:
                return {"success": False, "text": "", "stream_id": stream_id}

        material = str(material or "").strip()
        if not material:
            raise ValueError("material 不能为空")
        roster = None
        clean_names = [str(n).strip() for n in (names or []) if str(n).strip()]
        if clean_names:
            roster = self._roster_from_names(clean_names)
        ok, article = await self._generate_news(material, roster, self._agency_for_user(""))
        return {"success": ok, "text": article if ok else "", "stream_id": stream_id}

    # ── 首页卡片 ──

    @HomeCard(
        "obituary_newsroom_card",
        title="新闻三要素",
        description="把一段素材写成新闻体短文，并让三位虚构人物的死讯自然见报。",
        content=[
            {
                "type": "markdown",
                "content": "发送 `/新闻三要素 <素材>`，或用 `/新闻三要素` 引用一条消息，即可收到一篇新闻通稿。",
            },
            {
                "type": "list",
                "items": [
                    "/新闻三要素 <素材>：默认写固定那三人",
                    "/新闻三要素 <素材>（人名1，人名2，人名3）：改掉死亡名单",
                    "/新闻三要素 <素材>（甲，乙，丙，丁，戊）：多加名字就多死几个",
                    "/新闻三要素：引用消息后发送，用该消息当素材",
                    "/新闻三要素 重写：沿用上一份素材再出一版",
                    "/开头署名 观察者报：设置自己的新闻社（只对自己生效）",
                    "/开头署名 开 / 关：控制是否加「据○○社报道，」开头",
                    "对话里直接说“帮我按这段写篇新闻”也可以",
                ],
            },
            {
                "type": "markdown",
                "content": "所有人物死亡设定均为虚构玩梗，输出默认带声明。",
            },
        ],
        width="medium",
        order=200,
    )
    async def home_card(self) -> None:
        """声明 WebUI 首页卡片。"""

        return None

    # ── 生成主流程 ──

    async def _generate_news(
        self,
        material: str,
        roster: Optional[List[PersonConfig]] = None,
        agency: str = "",
        lead_in: Optional[bool] = None,
        from_image: bool = False,
    ) -> Tuple[bool, str]:
        """生成正文。

        先用提示词要一版；不合格（像思考链、缺人、体裁不对）就带上原因重写，
        最多 ``generation.max_attempts`` 次，全都不行才回退本地模板。

        Args:
            material: 用户提供的素材。
            roster: 本次指定的死亡名单；为空时用配置里的默认名单。
        """

        material = self._clean_material(material)
        attempts = max(1, int(self.config.generation.max_attempts))
        # 素材里已有死亡事件时用「替换死者」模式，不再另起转折句。
        replace_mode = _material_has_death(material)
        if replace_mode:
            self.ctx.logger.info("素材含死亡事件，改用替换模式")
        draft = ""

        for attempt in range(1, attempts + 1):
            if attempt == 1 or not draft:
                prompt = self._build_prompt(material, roster, from_image)
            else:
                prompt = self._build_repair_prompt(
                    material, draft, self._judge(draft, roster, replace_mode), roster, replace_mode
                )
            draft = await self._call_llm(prompt)
            problems = self._judge(draft, roster, replace_mode)
            if not problems:
                if attempt > 1:
                    self.ctx.logger.info("第 %s 次重写后正文合格", attempt)
                return True, self._finalize(draft, agency, lead_in)
            self.ctx.logger.warning("第 %s/%s 次正文不合格：%s", attempt, attempts, "；".join(problems))

        if not self.config.fallback.enabled:
            return False, ""
        self.ctx.logger.warning("模型多次未产出合格正文，改用本地模板")
        return True, self._finalize(self._fallback_news(material, roster), agency, lead_in)

    def _judge(
        self,
        draft: str,
        roster: Optional[List[PersonConfig]] = None,
        replace_mode: bool = False,
    ) -> List[str]:
        """检查一条候选正文，返回问题列表；空列表表示可用。

        这里防的是几类真实踩过的坑：
        - provider 把思考链当正文返回（低 token 预算下尤其常见）；
        - 模型只写了半截或干脆没写；
        - 漏掉当事人；
        - 输出的是提示词/规则复述，而不是成品；
        - 写成分段长文，而不是要求的一段话。
        """

        if not draft or not draft.strip():
            return ["正文为空"]

        problems: List[str] = []
        text = draft.strip()
        target = max(60, int(self.config.generation.target_length))
        limit = int(self.config.output.max_chars)

        if self._looks_like_reasoning(text):
            problems.append("内容看起来是思考过程/提纲，不是成品新闻，请直接写成通稿正文")

        missing = self._missing_people(text, roster)
        if missing:
            problems.append("缺少当事人：" + "、".join(self._resolve_missing(missing, roster)))

        body_len = len(re.sub(r"\s+", "", text))
        if body_len < int(target * 0.45):
            problems.append(f"正文过短（{body_len} 字），请写满约 {target} 字")
        if body_len > int(limit * 1.5):
            problems.append(f"正文过长（{body_len} 字），请压缩到 {limit} 字以内")

        # 只允许一段：正文里出现多个空行分段就算不合格。
        blank_separated = [seg for seg in re.split(r"\n\s*\n", text) if seg.strip()]
        if len(blank_separated) > 1:
            problems.append(f"写成了 {len(blank_separated)} 段，请压缩成一段话，不要分段")

        problems.extend(self._judge_structure(text, roster, replace_mode))

        return problems

    def _judge_structure(
        self,
        text: str,
        roster: Optional[List[PersonConfig]] = None,
        replace_mode: bool = False,
    ) -> List[str]:
        """检查正文结构。

        - 普通模式：要求「前半截素材延伸 — 转折词 — 后半截死亡名单」。
        - 替换模式（素材本身已有死亡事件）：**不要求**转折词，
          只要求名单里的人都写进正文、且不要凭空多出转折句。
        """

        pivot = str(self.config.generation.pivot or "值得注意的是").strip() or "值得注意的是"
        names = [person.name for person in _active_people(self.config.people, roster) if person.name]
        count = len(names)
        problems: List[str] = []

        if replace_mode:
            # 替换模式不要求转折词；出现了也提醒去掉——素材已有死亡，不需要再转折。
            if pivot in text:
                problems.append(
                    f"素材本身已经写了死亡事件，不要再用「{pivot}」另起一段死讯；"
                    "应直接把素材里的死者替换成名单里的人"
                )
            if len(re.sub(r"\s+", "", text)) < 30:
                problems.append("正文太短，请把素材经过写完整")
            return problems

        index = text.find(pivot)
        if index < 0:
            return [f"缺少转折词「{pivot}」：应在前半截写素材延伸，再用「{pivot}，」引出后半截的 {count} 人死讯"]

        head = text[:index]
        tail = text[index + len(pivot):]

        # 转折词前面必须是句号/分号之类的收束，不能用逗号硬接。
        stripped_head = head.rstrip()
        if stripped_head and stripped_head[-1] not in "。！？；…":
            problems.append(f"「{pivot}」前应先用句号或分号收住前一句，不要用逗号直接接转折词")

        if len(re.sub(r"\s+", "", head)) < 30:
            problems.append("转折词前面的素材延伸太短，请先交代背景、进展与影响再转折")

        if len(re.sub(r"\s+", "", tail)) < 30:
            problems.append(f"转折词后面内容太短，请在「{pivot}」之后完整写这 {count} 人的死讯")

        # 前半截不该出现死讯；出现了说明结构没守住。
        death_words = ("身亡", "离世", "死亡", "去世", "不治", "遇难", "丧生", "罹难")
        if any(word in head for word in death_words):
            problems.append(f"转折词前面不要写死讯，请把这 {count} 人的死亡全部放到「{pivot}」之后")

        # 名单里每个人都要落在后半截。
        tail_missing = [name for name in names if name and name not in tail]
        if tail_missing:
            problems.append(
                "这些当事人的死讯没有写在转折词之后：" + "、".join(tail_missing)
            )

        return problems

    @staticmethod
    def _looks_like_reasoning(text: str) -> bool:
        """判断这段文字是不是模型的思考过程或提纲，而非成品正文。

        依据：思考链的典型特征是大量英文元指令、自我提问、草稿标记，
        以及"标题：/结构：/字数"这类提纲式行首。
        """

        sample = text[:4000]
        lowered = sample.lower()

        # 中文思考口头禅 + 提纲式措辞
        chinese_tells = (
            "我们需要",
            "需要确保",
            "需要注意",
            "需要避免",
            "可以写",
            "可以称",
            "需要自然",
            "似乎",
            "？",
            "？",
        )
        # 英文元指令（中文正文里整段出现基本只可能是思考链）
        english_tells = (
            "need to",
            "ensure",
            "avoid",
            "let's draft",
            "title one line",
            "need mention",
            "use plain",
            "so total after",
        )
        marker_tells = ("**", "标题：", "结构：", "字数：", "结尾：", "电头：", "导语：", "正文：")

        chinese_hits = sum(1 for token in chinese_tells if token in sample)
        english_hits = sum(1 for token in english_tells if token in lowered)
        marker_hits = sum(1 for token in marker_tells if token in sample)

        # 成品新闻里不该出现成段的英文句子
        english_words = len(re.findall(r"[A-Za-z]{2,}", sample))
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", sample))
        english_ratio = english_words * 3 / max(1, english_words * 3 + chinese_chars)

        if english_hits >= 2 or english_ratio > 0.15:
            return True
        if marker_hits >= 2 and chinese_hits >= 3:
            return True
        # 问号极密 + 大量"需要/可以"，是自我推演
        if sample.count("？") + sample.count("?") >= 8 and chinese_hits >= 3:
            return True
        return chinese_hits >= 6

    # 素材来自图片时追加的指令。放在提示词里而不是素材里，
    # 否则模型会把指令当成素材内容抄进正文。
    _IMAGE_INSTRUCTION = (
        "【这条素材是对一张照片的画面描述】\n"
        "请把画面里呈现的场景当作**新闻现场**来写：人物、地点、动作按画面走，\n"
        "但要用新闻报道的口吻展开，补上合理的背景与各方反应。\n"
        "**正文里不要出现「图片」「照片」「画面」「图中」这类字样**，\n"
        "也不要写成「这张图里……」的描述句——读者应该以为你在报道一个真实发生的现场。\n"
    )

    _EMOJI_INSTRUCTION = (
        "【这条素材是一张表情包的标签，不是画面描述】\n"
        "标签代表的是**情绪与语境**，不是画面里真实存在的东西。\n"
        "请依据这些情绪基调，合理想象一个与之相符的新闻场景来写，\n"
        "**不要臆造具体人物的长相、穿着**，也不要写「表情包」「图片」这类字样。\n"
    )

    def _build_prompt(
        self,
        material: str,
        roster: Optional[List[PersonConfig]] = None,
        from_image: bool = False,
    ) -> str:
        gen = self.config.generation
        people = self.config.people
        extra = str(gen.extra_style or "").strip()
        extra_line = f"附加要求：{extra}" if extra else ""
        if from_image:
            image_block = (
                self._EMOJI_INSTRUCTION if self._last_material_kind == "emoji" else self._IMAGE_INSTRUCTION
            )
        else:
            image_block = ""
        # 素材里已经有死亡事件时，改成「替换死者」模式，不再另起转折句。
        template = _PROMPT_REPLACE if _material_has_death(material) else _PROMPT_RULES
        if _material_has_death(material):
            extra_line = f"10. {extra_line}" if extra_line else ""
        else:
            extra_line = f"11. {extra_line}" if extra_line else ""
        return template.format(
            material=material,
            target_length=int(gen.target_length),
            max_chars=int(self.config.output.max_chars),
            pivot=str(gen.pivot or "值得注意的是").strip() or "值得注意的是",
            people_count=len(_active_people(people, roster)),
            people_block=_format_people_block(people, roster),
            lock_block=_format_lock_block(people, roster),
            extra_style=extra_line,
            image_block=image_block,
        )

    def _build_repair_prompt(
        self,
        material: str,
        draft: str,
        problems: List[str],
        roster: Optional[List[PersonConfig]] = None,
        replace_mode: bool = False,
    ) -> str:
        """带上具体问题重写，比笼统说“不合格”有效得多。"""

        people = self.config.people
        count = len(_active_people(people, roster))
        reason = "\n".join(f"- {item}" for item in problems) or "- 请按硬性要求重写"
        pivot = str(self.config.generation.pivot or "值得注意的是").strip() or "值得注意的是"
        # 上一版是思考链/提纲时，不要把它塞回去，否则模型会顺着继续推演。
        looks_like_reasoning = self._looks_like_reasoning(draft)
        if looks_like_reasoning:
            draft_block = "（上一版输出的是思考过程，已丢弃，请从头直接写成品正文）"
        else:
            draft_block = f"【上一版】\n\"\"\"\n{draft[:2000]}\n\"\"\""

        if replace_mode:
            structure = (
                f"3. **不要用「{pivot}」这类转折结构**。素材里本来就写了死亡事件，\n"
                "   请顺着素材的叙事写下去，把素材里的死者直接换成名单里的人。\n"
                f"4. 把素材里的死者换成下列 {count} 人，每人全名各出现一次"
                "（不要漏人，也不要加名单外的人）：\n"
            )
            tail_rule = (
                "5. **保留素材原本的死因与场景**，不要改成别的事故原因；\n"
                f"{_format_lock_block(people, roster)}"
                "   各人身份按素材场景设定，互不相同。\n"
            )
        else:
            structure = (
                f"3. **这一段必须是「前—转折—后」两截，转折词固定为「{pivot}」：**\n"
                "   前半截只写素材的延伸（背景、进展、影响、各方反应），**不要提任何死讯**，"
                "   并以句号或分号收住；\n"
                f"   然后接「{pivot}，」，后半截集中写下列 {count} 人的死讯。\n"
                "   两截大致各占一半。\n"
                f"4. 下列 {count} 人的死讯都要写在后半截，每人全名各出现一次"
                "（不要漏人，也不要加名单外的人）：\n"
            )
            tail_rule = (
                "5. **各人身份必须按这条新闻的场景重新设定，不能沿用现实职务**；\n"
                f"{_format_lock_block(people, roster)}"
                "   各人身份互不相同，死因也各不相同，都要贴合素材场景。\n"
            )

        return (
            "上一次的输出不合格，问题如下：\n"
            f"{reason}\n\n"
            "请重新写一条中文新闻短讯。硬性要求：\n"
            "1. 直接输出成品，不要写任何分析、规划、提纲，不要复述要求，不要自问自答。\n"
            f"2. **全篇只有一段**，不分段、不空行、不用 Markdown；约 "
            f"{int(self.config.generation.target_length)} 字，"
            f"硬上限 {int(self.config.output.max_chars)} 字。\n"
            + structure
            + f"{_format_people_block(people, roster)}\n"
            + tail_rule
            + "6. 不要写清单、不要排比句，用通稿口吻。\n\n"
            f"【素材】\n\"\"\"\n{material}\n\"\"\"\n\n"
            f"{draft_block}\n\n"
            "现在只输出那一段正文本身。"
        )

    def _resolve_missing(self, missing: List[str], roster: Optional[List[PersonConfig]] = None) -> List[str]:
        """把校验结果还原成全名；姓氏重复时 ``missing`` 里放的本就是全名。"""

        resolved: List[str] = []
        for item in missing:
            person = self._person_by_surname(item, roster)
            if person is None:
                # 匹配不到人（配置刚改过、或本来就是全名）时原样保留，别吞掉。
                resolved.append(item)
                continue
            resolved.append(item if item == person.name else person.name)
        return resolved

    def _people_names(self, roster: Optional[List[PersonConfig]] = None) -> List[str]:
        """当前死亡名单的全名，用于重写提示与结构校验。"""

        return [person.name for person in self._people(roster) if person.name]

    async def _call_llm(self, prompt: str) -> str:
        """调用宿主模型；先按配置路由，失败后退回默认任务。"""

        gen = self.config.generation
        attempts: List[Dict[str, Any]] = []
        routed: Dict[str, Any] = {}
        if gen.task_name.strip():
            routed["task_name"] = gen.task_name.strip()
        if gen.model_name.strip():
            routed["model_name"] = gen.model_name.strip()
        if routed:
            attempts.append(routed)
        attempts.append({})

        for extra in attempts:
            try:
                result = await self.ctx.llm.generate(
                    prompt=prompt,
                    temperature=float(gen.temperature),
                    max_tokens=int(gen.max_tokens),
                    **extra,
                )
            except Exception as exc:  # 能力不可用、超时等
                self.ctx.logger.warning("LLM 调用异常：%s", exc)
                continue

            if not isinstance(result, dict):
                continue
            if not result.get("success", False):
                self.ctx.logger.warning("LLM 调用未成功：%s", result.get("error") or result)
                continue

            content = str(result.get("response") or "").strip()
            if not content:
                continue

            # 兜底：某些 provider 在思考链吃满预算时，会把 reasoning 原文塞进 response。
            # 两者完全相同时，这段一定是思考内容，不能当正文用。
            reasoning = str(result.get("reasoning") or "").strip()
            if reasoning and content == reasoning:
                self.ctx.logger.warning(
                    "模型返回的 response 与 reasoning 完全相同（疑似思考链被当作正文），已丢弃该结果"
                )
                continue
            return content

        return ""

    # ── 文本处理 ──

    @staticmethod
    def _clean_material(material: str) -> str:
        """压掉多余空白，避免超长素材把提示词撑爆。"""

        text = re.sub(r"[ \t]+", " ", str(material or ""))
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text[:6000]

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """模型偶尔会照抄 Markdown，这里统一清掉。"""

        cleaned = re.sub(r"```[a-zA-Z]*", "", text)
        cleaned = cleaned.replace("**", "").replace("__", "")
        cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"(?m)^\s{0,3}>\s?", "", cleaned)
        cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _finalize(self, article: str, agency: str = "", lead_in: Optional[bool] = None) -> str:
        """清洗成一条单段短讯：去 Markdown、压成一段、换结尾、按上限截断。

        默认不加标题前缀、不加声明、不加开头署名——
        ``output.headline_prefix``、``output.disclaimer``、``output.lead_in``
        都默认关闭，需要时在配置里打开。
        """

        text = self._strip_markdown(article)
        text = self._collapse_to_paragraph(text)

        # 统一收尾：把模型自己写的"仍在调查中"之类换成本插件的结尾句。
        if self.config.ending.enabled:
            text = self._apply_ending(text)

        # 开头署名（默认关闭）。
        want_lead = self.config.output.lead_in if lead_in is None else bool(lead_in)
        if want_lead:
            text = self._apply_lead_in(text, agency)

        prefix = str(self.config.output.headline_prefix or "").strip()
        if prefix and not text.startswith(prefix):
            text = f"{prefix}{text}"

        text = self._truncate_keep_ending(text, int(self.config.output.max_chars))

        disclaimer = str(self.config.output.disclaimer_text or "").strip()
        if self.config.output.disclaimer and disclaimer and disclaimer not in text:
            text = f"{text}\n\n{disclaimer}"

        return text

    def _apply_lead_in(self, text: str, agency: str = "") -> str:
        """在正文开头加「据○○社报道，」，○○ 取自 ``output.agency``。

        模型可能自己也写了「据……报道」，先剥掉再统一加，避免出现两句署名。
        """

        body = str(text or "").strip()
        if not body:
            return body

        agency = str(agency or "").strip() or str(self.config.output.agency or "盖金社").strip() or "盖金社"

        # 去掉正文开头已有的电头/署名，避免出现两句：
        # 「据……报道，」「本报讯，」「本报讯（记者××），」「××社×月×日电，」等。
        # 括号部分必须整组可选，不能写成 [（(]?[^）)]{0,20}——那会把正文一起吃掉。
        body = re.sub(
            r"^\s*(?:"
            r"据[^，,。]{1,20}报道[，,]?"
            r"|本报讯(?:[（(][^）)]{0,20}[）)])?[，,]?"
            r"|[^，,。]{1,12}(?:社|报)\s*\d{0,2}月\d{0,2}日[电讯][，,]?"
            r")",
            "",
            body,
        ).lstrip()

        head = f"据{agency}报道，"
        if body.startswith(head):
            return body
        return f"{head}{body}"

    # ── 每个用户自己的「新闻社」名 ──

    def _agency_file(self) -> Optional[Path]:
        """用户自定义新闻社的持久化文件路径；拿不到 data_dir 时返回 None。"""

        try:
            data_dir = Path(self.ctx.paths.data_dir)
        except Exception:
            return None
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        return data_dir / "user_agency.json"

    def _load_user_profiles(self) -> Dict[str, Dict[str, Any]]:
        """读取用户设置（user_id -> {name, lead_in}）。

        兼容两种历史格式：
        - 旧：``{"10001": "观察者报"}``（只存了名称）
        - 新：``{"10001": {"name": "观察者报", "lead_in": true}}``
        """

        path = self._agency_file()
        if path is None or not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.ctx.logger.warning("读取用户设置失败：%s", exc)
            return {}
        if not isinstance(raw, dict):
            return {}

        profiles: Dict[str, Dict[str, Any]] = {}
        for key, value in raw.items():
            uid = str(key).strip()
            if not uid:
                continue
            if isinstance(value, str):
                # 旧格式：字符串即名称
                name = value.strip()
                if name:
                    profiles[uid] = {"name": name}
                continue
            if isinstance(value, dict):
                entry: Dict[str, Any] = {}
                name = str(value.get("name") or "").strip()
                if name:
                    entry["name"] = name
                if "lead_in" in value:
                    entry["lead_in"] = bool(value.get("lead_in"))
                if entry:
                    profiles[uid] = entry
        return profiles

    def _save_user_profiles(self, profiles: Dict[str, Dict[str, Any]]) -> bool:
        """写回用户设置。"""

        path = self._agency_file()
        if path is None:
            return False
        cleaned = {uid: dict(value) for uid, value in profiles.items() if uid and value}
        try:
            path.write_text(
                json.dumps(cleaned, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except Exception as exc:
            self.ctx.logger.error("保存用户设置失败：%s", exc)
            return False

    def _agency_for_user(self, user_id: str) -> str:
        """取该用户生效的新闻社名；没自定义就用全局配置。"""

        default = str(self.config.output.agency or "盖金社").strip() or "盖金社"
        uid = str(user_id or "").strip()
        if not uid:
            return default
        profile = self._load_user_profiles().get(uid) or {}
        return str(profile.get("name") or "").strip() or default

    def _lead_in_for_user(self, user_id: str) -> bool:
        """该用户是否要「据○○社报道」开头；没设置就用全局配置。"""

        uid = str(user_id or "").strip()
        if not uid:
            return bool(self.config.output.lead_in)
        profile = self._load_user_profiles().get(uid) or {}
        if "lead_in" in profile:
            return bool(profile["lead_in"])
        return bool(self.config.output.lead_in)

    def _pick_verb(self) -> str:
        """从 ``ending.verbs`` 里随机挑一个词；空列表回落到「作哀」。"""

        verbs = [str(v).strip() for v in (self.config.ending.verbs or []) if str(v).strip()]
        if not verbs:
            return "作哀"
        return random.choice(verbs)

    def _render_ending(self) -> str:
        """按模板渲染结尾句，词位随机填充。"""

        cfg = self.config.ending
        verb = self._pick_verb()
        template = str(cfg.template or "").strip() or "让我们为他们{verb}"
        ending = template.replace("{verb}", verb)
        ending = ending.replace("｛verb｝", verb).replace("{{verb}}", verb)

        # 模板里没写 emoji 时按开关补一个，避免用户漏配。
        if cfg.use_emoji and not re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", ending):
            ending = ending.rstrip() + "🙏"
        return ending.strip()

    # 模型爱写的收尾套话：只要整句里含这些字样，就整句剥掉换成插件结尾。
    _TAIL_KEYWORDS = (
        "仍在进一步调查",
        "仍在调查",
        "正在进一步调查",
        "正在调查",
        "有待进一步调查",
        "尚待进一步调查",
        "仍在进一步核实",
        "仍在核实",
        "正在核实",
        "有待进一步核实",
        "尚待进一步核实",
        "仍在进一步处理",
        "具体原因仍在调查",
        "具体原因尚待",
        "原因仍在调查",
        "相关情况仍在",
        "相关情况正在",
    )

    def _apply_ending(self, text: str) -> str:
        """去掉模型自带的"仍在调查中"式收尾，换上配置的结尾句。"""

        body = str(text or "").strip()
        if not body:
            return body

        ending = self._render_ending()

        # 逐句检查尾部：最后一句命中套话就丢掉，最多剥 3 句。
        # 只要剥完还剩 2 句以上就允许，避免短正文被误判成"不能剥"。
        for _ in range(3):
            sentences = re.findall(r"[^。！？!?…]+[。！？!?…]?", body)
            if len(sentences) < 2:
                break
            last = sentences[-1]
            if not any(word in last for word in self._TAIL_KEYWORDS):
                break
            candidate = body[: len(body) - len(last)].rstrip()
            remaining = re.findall(r"[^。！？!?…]+[。！？!?…]?", candidate)
            if len(remaining) < 2 or len(re.sub(r"\s+", "", candidate)) < 20:
                break
            body = candidate

        body = body.rstrip("。！？!?… 　")
        if not body:
            return ending
        return f"{body}。{ending}"

    def _truncate_keep_ending(self, text: str, limit: int) -> str:
        """按上限截断正文，但**保住结尾的哀悼句**。

        直接截断会把「让我们为他们哀悼🙏」切掉，或把 🙏 切成半个字符，
        所以这里把结尾句拆出来，只用剩余额度截断主体，再拼回去。

        注意：这里不能重新渲染结尾句——词是随机的，重渲染会得到不同的词，
        因而认不出上一句是结尾。改为从文本尾部直接找 🙏 定位。
        """

        if limit <= 0 or len(text) <= limit:
            return text
        if not self.config.ending.enabled:
            return self._truncate(text, limit)

        # 结尾句以 🙏 收尾：从最后一个 🙏 往前找句号，切出结尾。
        pos = text.rfind("🙏")
        if pos < 0 or pos < len(text) - 4:
            return self._truncate(text, limit)
        cut = text.rfind("。", 0, pos)
        if cut < 0:
            return self._truncate(text, limit)

        body = text[:cut]
        ending = text[cut + 1 :]
        room = limit - len(ending) - 1
        if room <= 0:
            return ending
        return f"{self._truncate(body, room).rstrip('。！？!?… 　')}。{ending}"

    @staticmethod
    def _collapse_to_paragraph(text: str) -> str:
        """把多段文本压成一段：段落之间直接拼接，不保留空行。"""

        parts = [seg.strip() for seg in re.split(r"\n\s*\n", text) if seg.strip()]
        if not parts:
            return ""
        merged = ""
        for part in parts:
            # 段内多余换行也一并去掉，避免聊天窗口里被切成多行。
            line = re.sub(r"\n+", "", part).strip()
            if not line:
                continue
            if merged and not merged.endswith(("。", "！", "？", "…", "”", "」", "）")):
                merged += "。"
            merged += line
        return merged.strip()

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """按句子边界截断，尽量不切在半句话上。"""

        if limit <= 0 or len(text) <= limit:
            return text
        window = text[:limit]
        cut = max(window.rfind("。"), window.rfind("！"), window.rfind("？"), window.rfind("\n"))
        if cut < limit * 0.6:
            return window.rstrip() + "……"
        return window[: cut + 1].rstrip()

    # ── 当事人校验 ──

    def _people(self, roster: Optional[List[PersonConfig]] = None) -> List[PersonConfig]:
        """当前生效的死亡名单（可被命令里临时指定的人名覆盖）。"""

        return _active_people(self.config.people, roster)

    @staticmethod
    def _surname(name: str) -> str:
        return str(name or "").split("·")[-1].strip()

    def _person_by_surname(self, surname: str, roster: Optional[List[PersonConfig]] = None) -> Optional[PersonConfig]:
        for person in self._people(roster):
            if self._surname(person.name) == surname:
                return person
        return None

    def _missing_people(self, article: str, roster: Optional[List[PersonConfig]] = None) -> List[str]:
        """返回正文里没被提到的当事人姓氏。"""

        people = self._people(roster)
        if not article:
            return [self._surname(person.name) for person in people]
        surnames = [self._surname(person.name) for person in people]
        if len(set(surnames)) < len(surnames):
            # 姓氏重复时退化为全名匹配，避免误判。
            return [person.name for person in people if person.name not in article]
        return [surname for surname in surnames if surname and surname not in article]

    # ── 本地模板回退 ──

    def _fallback_news(self, material: str, roster: Optional[List[PersonConfig]] = None) -> str:
        """不依赖模型也能产出的一段话短讯。

        身份按素材关键词粗分场景：战争 -> 军官，事故 -> 工班长，体育 -> 教练……
        没匹配上就用通用中性身份，避免硬套“制作人”这种现实职务。
        """

        cfg = self.config.fallback
        excerpt = self._material_excerpt(material)
        org = str(cfg.org_name or "有关部门").strip()

        roles, causes = self._scene_roles(material)
        members = self._people(roster)
        if not members:
            return ""

        # 用素材哈希挑开场措辞，同一素材结果稳定，不同素材不至于完全一样。
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        lead = [
            "据当地通报",
            "综合多方消息",
            "据现场人员介绍",
        ][int(digest[:2], 16) % 3]
        follow = [
            f"{org}表示，相关处置仍在进行中，现场秩序已逐步恢复。",
            f"截至发稿，{org}已介入处置，后续安排仍在协调。",
            f"{org}称，事件的具体原因与损失情况仍在核查。",
        ][int(digest[4:6], 16) % 3]
        pivot = str(self.config.generation.pivot or "值得注意的是").strip() or "值得注意的是"

        # 后半截逐人成句；身份与死因超出预设时循环取用，保证任意人数都写得出。
        clauses: list[str] = []
        for index, person in enumerate(members):
            title = str(person.title or "").strip() or roles[index % len(roles)]
            cause = str(person.cause_hint or "").strip() or causes[index % len(causes)]
            clauses.append(f"{title}{person.name}{cause}")

        if _material_has_death(material):
            # 素材本身已有死亡事件：不另起转折，直接把死者换成名单里的人。
            text = (
                f"{lead}，与“{excerpt}”有关的事件已造成人员伤亡，{follow}"
                + "；".join(clauses)
                + f"。上述 {len(members)} 人的身份已由身边人员确认。"
            )
        else:
            # 前半截只写素材延伸，不出现任何死讯；后半截从转折词开始写死亡名单。
            text = (
                f"{lead}，与“{excerpt}”有关的事件连日引发关注，相关处置与核查仍在推进。"
                f"{follow}"
                f"{pivot}，" + "；".join(clauses) + "。"
                f"上述 {len(members)} 人的身份已由身边人员确认。"
            )
        # 收尾统一交给 _finalize（按 ending 配置渲染），模板这里不写死。
        return text

    @staticmethod
    def _scene_roles(material: str) -> Tuple[List[str], List[str]]:
        """按素材关键词粗略判断场景，返回身份池与死因池。

        Returns:
            Tuple[List[str], List[str]]: ``(身份池, 死因池)``，两者等长，可循环取用。
        """

        text = str(material or "")
        scenes = [
            (
                ("比赛", "球", "联赛", "运动", "教练", "冠军", "夺冠", "决赛", "赛季", "球队", "篮球", "足球"),
                ["带队教练", "队医", "随队翻译", "器材管理员", "助理教练", "青训主管"],
                [
                    "在场边突发心脏骤停离世",
                    "在随队途中遭遇车祸身亡",
                    "在赛后返程中突发疾病身亡",
                    "在整理器材时发生意外身亡",
                    "在带队训练后突发疾病离世",
                    "在出差考察途中突发状况身亡",
                ],
            ),
            (
                (
                    "战争", "战事", "战场", "战线", "交火", "军队", "军事", "士兵",
                    "部队", "冲突", "空袭", "炮击", "前线", "导弹", "武装",
                ),
                ["某部指挥官", "作战参谋", "前线通信兵", "后勤补给兵", "战地医护员", "工兵排长"],
                [
                    "在阵地转移途中遭炮击身亡",
                    "在指挥所遇袭中身亡",
                    "在执行通信保障任务时身亡",
                    "在运送补给途中触雷身亡",
                    "在抢救伤员时遇袭身亡",
                    "在排雷作业中殉职",
                ],
            ),
            (
                ("事故", "爆炸", "坍塌", "火灾", "工厂", "矿", "泄漏", "车间"),
                ["车间主管", "夜班班长", "设备巡检员", "安全员", "消防值班员", "仓库管理员"],
                [
                    "在抢险中被倒塌构件击中身亡",
                    "在连续处置后突发心脏骤停离世",
                    "在巡查泄漏点时中毒身亡",
                    "在组织撤离时吸入浓烟身亡",
                    "在扑救初期火灾时被爆燃波及身亡",
                    "在清点物资时被坠物砸中身亡",
                ],
            ),
            (
                ("车", "交通", "铁路", "航班", "运输", "司机", "道路"),
                ["车队调度员", "长途驾驶员", "线路检修工", "站点值班员", "押运员", "信号维护员"],
                [
                    "在赶赴现场途中遭遇车祸身亡",
                    "在连续驾驶后突发心梗离世",
                    "在作业时被列车撞击身亡",
                    "在站台疏导时意外坠亡",
                    "在押运途中车辆翻覆身亡",
                    "在检修信号设备时触电身亡",
                ],
            ),
            (
                ("科研", "实验", "医院", "病", "疫苗", "药"),
                ["课题组负责人", "主治医师", "设备工程师", "实验员", "护士长", "质检员"],
                [
                    "在实验室连夜值守后突发疾病离世",
                    "在连续值守后因并发症离世",
                    "在检修设备时意外触电身亡",
                    "在处置样本时意外暴露身亡",
                    "在连班后突发疾病离世",
                    "在抽检作业中意外身亡",
                ],
            ),
        ]
        for keywords, roles, causes in scenes:
            if any(word in text for word in keywords):
                return roles, causes

        # 兜底：不假设任何行业。
        return (
            ["现场负责人", "值班主管", "随行工作人员", "现场协调员", "后勤保障员", "安全巡查员"],
            [
                "在处置现场突发疾病离世",
                "在赶赴现场途中遭遇事故身亡",
                "在值守期间突发状况离世",
                "在协调转运时意外身亡",
                "在搬运物资时被坠物砸中身亡",
                "在夜间巡查时发生意外身亡",
            ],
        )

    @staticmethod
    def _material_excerpt(material: str) -> str:
        text = re.sub(r"\s+", " ", str(material or "")).strip()
        if len(text) <= 40:
            return text or "近期的一项工作"
        return text[:40].rstrip("，,。. ") + "…"

    # ── 发送 ──

    async def _deliver(self, article: str, stream_id: str, agency: str = "") -> bool:
        """发送正文。

        **默认走合并转发**（``send.forward``），而不是普通文本。原因：

        带出站拦截的插件（如「发言频率控制」）会在 ``send_service.before_send``
        里取消「看起来像麦麦闲聊」的纯文本，命令结果也常被误杀。
        但那条闸门对**非文本出站**（image / emoji / forward / file …）一律放行——
        合并转发正属于这一类，所以走它能稳定送达。

        其它注意：
        - 宿主在 ``set_reply=True`` 且没有 ``reply_message`` 时会直接丢弃整条消息，
          插件拿不到被回复消息对象，所以一律不传 ``set_reply``。
        - 没有可用会话 ID 时直接失败，不拿可疑值去发。

        Returns:
            bool: 是否确认发送成功。
        """

        # 这一层兜底：绝不用可疑的会话 ID 去发消息。
        stream_id = self._sanitize_stream_id(stream_id)
        if not stream_id:
            self.ctx.logger.error("没有可用的会话 ID，正文无法送达")
            return False

        cfg = self.config.output
        nickname = str(agency or "").strip() or str(cfg.forward_nickname or "盖金社")
        mode = str(getattr(cfg, "delivery_mode", "text_first") or "text_first").strip().lower()

        async def _send_forward() -> bool:
            message = {
                "user_id": "0",
                "nickname": nickname,
                "segments": [{"type": "text", "content": article}],
            }
            return bool(await self.ctx.send.forward([message], stream_id))

        async def _send_text() -> bool:
            return bool(await self.ctx.send.text(article, stream_id))

        # 按配置决定先后顺序：
        #   text_first（默认）：先直接发文本，被出站拦截再退回合并转发——
        #                       群里能直接看到正文，不是聊天记录卡片。
        #   forward_first     ：先合并转发，稳定绕开拦截。
        #   text / forward    ：只用其中一种。
        if mode == "text":
            sent = await _send_text()
        elif mode == "forward":
            sent = await _send_forward()
        elif mode == "forward_first":
            sent = await _send_forward()
            if not sent:
                self.ctx.logger.info("合并转发未成功，改为直接发文本")
                sent = await _send_text()
        else:  # text_first
            sent = await _send_text()
            if not sent:
                self.ctx.logger.info("直接发送被拦，改用合并转发兜底")
                sent = await _send_forward()

        if not sent:
            self.ctx.logger.error(
                "正文发送失败：stream=%s length=%s。若宿主的频率控制插件拦了这条，"
                "把「新闻三要素」加到它的「额外命令触发词」里，或关闭它的「启用出站拦截」",
                stream_id,
                len(article),
            )
        return bool(sent)

    # ── 输入辅助 ──

    @staticmethod
    def _has_quote(message: Any, text: str) -> bool:
        """判断这条命令到底有没有引用消息。"""

        if isinstance(message, dict):
            if str(message.get("reply_to") or "").strip():
                return True
            for segment in message.get("raw_message") or []:
                if isinstance(segment, dict) and str(segment.get("type") or "").lower() == "reply":
                    return True
        raw = str(text or "")
        # 用普通子串判断，避免方括号正则转义踩坑。
        return ("Reply(" in raw) or ("[回复了" in raw) or ("[引用" in raw)

    @staticmethod
    def _material_from_command_text(text: str) -> str:
        raw = str(text or "").strip()
        match = re.match(COMMAND_PATTERN, raw)
        if match:
            return str(match.group("material") or "").strip()
        return ""

    # 从 ``Reply(...)`` 里抠 target_content；
    # 引用时宿主把引用段拼成了这段结构化文本。
    _REPLY_CONTENT_RE = re.compile(r"target_content=(?P<q>['\"])(?P<content>.*?)\1", re.S)
    # 采到真正的结尾括号为止：被引用的图片内容里可能还套着 "[图片：xxx]"
    _BRACKET_QUOTE_RE = re.compile(r"\[回复了[^\]：:]*[：:]\s*(?P<content>[\s\S]+)\]\s*(?:$|[/／])", re.S)

    @classmethod
    def _quoted_text_from_command(cls, text: str) -> str:
        """从命令原文的引用前缀里提取被引用消息的文字。

        引用消息时 ``processed_plain_text`` 形如::

            Reply(target_message_id='1', target_sender='小王',
                  target_content='昨晚空调坏了') /新闻三要素

        或老格式::

            [回复了小王的旧消息: 昨晚空调坏了] /新闻三要素

        这里把 ``target_content`` 或方括号里的正文抠出来当素材。
        """

        raw = str(text or "")
        if not raw:
            return ""

        match = cls._REPLY_CONTENT_RE.search(raw)
        if match:
            content = str(match.group("content") or "").strip()
            # 去掉 Quote(...) 之类的包裹，保留里面真正的文字
            content = re.sub(r"^\s*Quote\(|>\)*\s*$", "", content).strip()
            if content:
                return content

        match = cls._BRACKET_QUOTE_RE.search(raw)
        if match:
            content = str(match.group("content") or "").strip()
            if content:
                return content

        return ""

    # 引用前缀被 Hook 挪到命令末尾后，素材里会残留这一段，需要剥掉。
    # 注意：[回复了…] 的内容里常带方括号（引用的正是图片或表情包时，
    # 内容是 "[图片：…]" / "[emoji]"），所以不能用 [^\]]* 偷懒——
    # 那样会在内容里的第一个 ] 处截断，只剩一个 "]"。
    _TRAILING_QUOTE_RE = re.compile(
        r"\s*(?:Reply\([^()]*\)|\[回复了[\s\S]*\]|\[引用[\s\S]*\])\s*$",
        re.IGNORECASE,
    )

    @classmethod
    def _strip_quote_prefix(cls, text: str) -> str:
        """去掉素材里残留的引用前缀（可能在开头，也可能在末尾）。"""

        value = str(text or "")
        # 末尾的（Hook 归位后会出现在这里）
        previous = None
        while previous != value:
            previous = value
            value = cls._TRAILING_QUOTE_RE.sub("", value).rstrip()
        # 开头的（未归位时的原生形态）
        match = cls._LEADING_QUOTE_PREFIX_MATERIAL.match(value)
        if match:
            value = value[match.end():].lstrip()
        return value.strip()

    _LEADING_QUOTE_PREFIX_MATERIAL = re.compile(
        r"^(?:\s*(?:Reply\([^()]*\)|\[回复了[\s\S]*?\](?=\s|$)|\[引用[\s\S]*?\](?=\s|$)))+",
        re.IGNORECASE,
    )

    # ── 从命令里拆出「素材 + 自定义人名」 ──

    @staticmethod
    def _split_material_and_names(raw: str) -> Tuple[str, List[str]]:
        """把命令正文拆成「素材」和「括号里的人名列表」。

        用括号明确框住名单，彻底避免和素材正文混淆：
        ``/新闻三要素 美国大选 （特朗普，奥巴马，拜登）``
        → 素材「美国大选」，名单 [特朗普, 奥巴马, 拜登]。

        支持半角 ``()`` 与全角 ``（）``，名字之间用中英文逗号、顿号、分号或空格分隔。

        Returns:
            Tuple[str, List[str]]: ``(素材, 人名列表)``；没写括号时人名为空。
        """

        text = str(raw or "").strip()
        if not text:
            return "", []

        # 取**最后一对**括号作为名单，前面全部算素材（素材里可以出现别的括号）。
        pattern = re.compile(r"[(（]([^()（）]*)[)）]\s*$")
        match = pattern.search(text)
        if not match:
            return text, []

        inner = match.group(1)
        material = text[: match.start()].strip()

        names: List[str] = []
        for part in re.split(r"[,，、;；/|]+|\s+", inner):
            name = part.strip().strip("、,，;；")
            if not name:
                continue
            if len(name) > _NAME_MAX_LEN:
                continue
            if "。！？!?…：:" in name:
                continue
            names.append(name)

        if not names:
            # 括号里没内容：整句都算素材，不要把括号当名单。
            return text, []

        # 只有括号、没有素材时也返回名单（素材为空），
        # 让「引用消息 + 只给名单」这种用法能从引用里取素材。
        return material, names

    # 宿主把图片转成 "[图片：xxx]" / "[表情包：xxx]" 塞进文本，
    # 引用图片当素材时会带这层壳。下面两个正则用来认出它。
    _IMAGE_MATERIAL_RE = re.compile(
        r"^\s*[\[【]\s*(?:图片|图像|照片|image|img)\s*[:：]\s*(?P<desc>.+?)\s*[\]】]\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    _EMOJI_MATERIAL_RE = re.compile(
        r"^\s*[\[【]\s*(?:表情包|emoji)\s*[:：]\s*(?P<tags>.+?)\s*[\]】]\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    _IMAGE_INLINE_RE = re.compile(
        r"[\[【]\s*(?:图片|图像|照片|表情包)\s*[:：]\s*(?P<desc>[^\]】]+)\s*[\]】]"
    )

    # 只有标记、没有实际内容的空壳（宿主识别失败或未做视觉识别时会这样）
    _EMPTY_SHELL_RE = re.compile(
        r"^\s*[\[【［(（]?\s*"
        r"(?:emoji|image|img|图片|图像|照片|表情包|动图|视频|语音|file|文件)"
        r"\s*[\]】］)）]?\s*$",
        re.IGNORECASE,
    )

    # 宿主有时来不及在消息入库前跑完图片识别：
    # processed_plain_text 是空的，引用时只剩占位符 ``[image]``，
    # 但描述其实已经补进了 images 表。这里去把它捞回来。
    _IMAGE_HASH_RE = re.compile(r"\bhash\b['\"]?\s*[:=]?\s*['\"]?(?P<hash>[0-9a-f]{32,128})", re.IGNORECASE)

    async def _image_description_by_hash(self, image_hash: str) -> str:
        """按图片 hash 查 images 表拿画面描述。"""

        value = str(image_hash or "").strip()
        if not value:
            return ""
        query = getattr(self.ctx.db, "query", None)
        if not callable(query):
            return ""
        try:
            result = await query(
                model_name="Images",
                query_type="get",
                filters={"image_hash": value},
                limit=1,
                single_result=True,
            )
        except Exception as exc:
            self.ctx.logger.warning("按 hash 查图片描述失败：%s", exc)
            return ""
        if not isinstance(result, dict):
            return ""
        candidates = result.get("data") or result.get("result") or []
        if isinstance(candidates, dict):
            candidates = [candidates]
        for item in candidates if isinstance(candidates, list) else []:
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description") or "").strip()
            if desc:
                return desc
        return ""

    async def _image_record_by_hash(self, image_hash: str) -> Dict[str, Any]:
        """按 hash 取整条 images 记录（含 image_type）。"""

        value = str(image_hash or "").strip()
        if not value:
            return {}
        query = getattr(self.ctx.db, "query", None)
        if not callable(query):
            return {}
        try:
            result = await query(
                model_name="Images",
                query_type="get",
                filters={"image_hash": value},
                limit=1,
                single_result=True,
            )
        except Exception as exc:
            self.ctx.logger.warning("按 hash 查图片记录失败：%s", exc)
            return {}
        if not isinstance(result, dict):
            return {}
        candidates = result.get("data") or result.get("result") or {}
        if isinstance(candidates, list):
            candidates = candidates[0] if candidates else {}
        return candidates if isinstance(candidates, dict) else {}

    async def _recover_image_material(self, message: Any, stream_id: str) -> str:
        """引用图片但正文为空时，从 images 表把描述补回来。"""

        if not isinstance(message, dict):
            return ""
        # 1) 从 raw_message 的 reply 段拿被引用消息 ID
        target_id = ""
        for segment in message.get("raw_message") or []:
            if not isinstance(segment, dict):
                continue
            if str(segment.get("type") or "").lower() != "reply":
                continue
            data = segment.get("data")
            if isinstance(data, dict):
                target_id = str(data.get("target_message_id") or "").strip()
            if target_id:
                break
        if not target_id:
            target_id = str(message.get("reply_to") or "").strip()
        if not target_id:
            self.ctx.logger.warning("图片恢复：拿不到被引用消息 ID")
            return ""

        # 2) 取被引用消息的原始结构，找图片 hash
        get_by_id = getattr(self.ctx.message, "get_by_id", None)
        if not callable(get_by_id):
            self.ctx.logger.warning("图片恢复：ctx.message.get_by_id 不可用")
            return ""
        try:
            record = await get_by_id(target_id, chat_id=stream_id)
        except Exception as exc:
            self.ctx.logger.warning("图片恢复：取被引用消息失败 id=%s err=%s", target_id, exc)
            return ""
        if not isinstance(record, dict):
            self.ctx.logger.warning(
                "图片恢复：被引用消息取不回来 id=%s 返回类型=%s", target_id, type(record).__name__
            )
            return ""

        # 宿主 message.get_by_id 返回的是包了一层的结构：
        #   {"success": True, "message": {...实际消息...}}
        # 直接拿外层会取不到 raw_message。
        inner = record.get("message")
        if isinstance(inner, dict):
            record = inner
        elif record.get("success") is False:
            self.ctx.logger.warning(
                "图片恢复：get_by_id 失败 id=%s err=%s", target_id, record.get("error")
            )
            return ""
        hashes: List[str] = []
        for segment in record.get("raw_message") or []:
            if not isinstance(segment, dict):
                continue
            if str(segment.get("type") or "").lower() not in {"image", "emoji"}:
                continue
            # 宿主序列化时 hash 在**段的顶层**：
            #   {"type": "image", "data": <content>, "hash": <binary_hash>}
            # （见 PluginMessageUtils 的 ImageComponent / EmojiComponent 分支），
            # 不在 data 里。data 通常是 content 字符串。
            h = str(segment.get("hash") or "").strip()
            if not h:
                data = segment.get("data")
                if isinstance(data, dict):
                    h = str(data.get("hash") or "").strip()
            if h:
                hashes.append(h)
        if not hashes:
            self.ctx.logger.warning(
                "图片恢复：被引用消息里没有图片 hash，raw 段类型=%s",
                [str(s.get("type")) for s in (record.get("raw_message") or []) if isinstance(s, dict)],
            )
            return ""

        # 3) 按 hash 查描述
        db_query = getattr(getattr(self.ctx, "db", None), "query", None)
        if not callable(db_query):
            self.ctx.logger.warning("图片恢复：ctx.db.query 不可用（缺少 database.query 能力？）")
            return ""
        for h in hashes:
            record_data = await self._image_record_by_hash(h)
            if not record_data:
                continue
            desc = str(record_data.get("description") or "").strip()
            image_type = str(record_data.get("image_type") or "").strip().upper()
            if not desc:
                continue
            if image_type == "EMOJI":
                # 表情包只有一串标签（如「大笑,笑死,刷手机」），没有画面描述。
                # 直接拿标签当画面会很怪，但完全丢掉又浪费了语义。
                # 折中：把标签转成一句"这是一张表达××情绪的表情图"，让模型据此起稿。
                tags = [t.strip() for t in desc.replace("，", ",").split(",") if t.strip()]
                if not tags:
                    continue
                self.ctx.logger.info("从 images 表补回表情包标签：%s", tags)
                return f"[表情包：{'、'.join(tags)}]"
            self.ctx.logger.info("从 images 表补回图片描述（长度=%s）", len(desc))
            return f"[图片：{desc}]"
        self.ctx.logger.warning("图片恢复：images 表里这些 hash 都没有可用描述：%s", [h[:12] for h in hashes])
        return ""

    @classmethod
    def _is_empty_shell(cls, material: str) -> bool:
        """素材是否只是个空壳标记（没有任何可写的内容）。

        宿主对表情包通常只写 ``[emoji]`` / ``[表情包]``，
        既没有画面描述也没有标签，这种素材拿去写只会让模型凭空编。
        """

        text = str(material or "").strip()
        if not text:
            return True
        # 先剥掉引用前缀，再看剩下的是不是纯标记
        stripped = cls._strip_quote_prefix(text)
        if not stripped:
            return True
        if cls._EMPTY_SHELL_RE.match(stripped):
            return True
        # "[表情包: 呆萌,可爱]" 这种带标签的，虽然不能当画面用，但至少有语义
        return False

    def _normalize_material(self, material: str) -> Tuple[str, bool]:
        """把图片/表情包素材展开成可写内容，并记录来源类型到 ``_last_material_kind``。"""

        text, is_image = type(self)._normalize_material_impl(material)
        if is_image:
            lowered = str(material or "").lower()
            self._last_material_kind = (
                "emoji" if ("表情包" in str(material or "") or "emoji" in lowered) else "image"
            )
        return text, is_image

    @classmethod
    def _normalize_material_impl(cls, material: str) -> Tuple[str, bool]:
        """把「图片素材」展开成可写的事件描述。

        引用一张图片发命令时，素材实际是 ``[图片：一只橘猫趴在键盘上]``。
        直接把这段壳塞进新闻会很怪，所以这里剥掉外层标记，
        并补一句"据画面显示"之类的引导语，让模型按"看到了一张图"来写，
        而不是把它当成有人说过的一句话。

        Returns:
            Tuple[str, bool]: ``(处理后的素材, 是否来自图片)``。
        """

        text = str(material or "").strip()
        if not text:
            return "", False

        # 表情包只有标签，没有画面描述，必须走单独分支（先于图片判断）。
        emoji_match = cls._EMOJI_MATERIAL_RE.match(text)
        if emoji_match:
            tags = str(emoji_match.group("tags") or "").strip()
            if tags:
                # 只返回标签本身，说明性文字放进提示词（见 _EMOJI_INSTRUCTION），
                # 否则模型会把说明当素材抄进正文。
                return tags, True

        match = cls._IMAGE_MATERIAL_RE.match(text)
        if match:
            desc = str(match.group("desc") or "").strip()
            if desc:
                # 只把「画面描述」本身当素材。
                # 写作要求必须放到提示词的指令区，否则模型会把要求原样抄进正文
                # （实测出现过「与『…请把画面里呈现的场景当作新闻现场来写』有关」这种句子）。
                return desc, True

        # 素材里夹着图片描述（图文混排的引用），把标记就地展开
        if cls._IMAGE_INLINE_RE.search(text):
            def _repl(m: "re.Match[str]") -> str:
                desc = str(m.group("desc") or "").strip()
                # 同样只留画面本身，不加任何指令性文字。
                return desc if desc else ""

            expanded = cls._IMAGE_INLINE_RE.sub(_repl, text)
            expanded = re.sub(r"\s{2,}", " ", expanded).strip()
            if expanded and expanded != text:
                return expanded, True

        return text, False

    def _roster_from_names(self, names: List[str]) -> List[PersonConfig]:
        """按命令里给的人名造临时名单；能对上配置的沿用其身份设定。"""

        roster: List[PersonConfig] = []
        known = {
            str(person.name or "").strip(): person
            for person in list(getattr(self.config.people, "list", []) or [])
            if str(person.name or "").strip()
        }
        for name in names:
            matched = known.get(name)
            roster.append(
                PersonConfig(
                    name=name,
                    title=str(matched.title or "").strip() if matched else "",
                    cause_hint=str(matched.cause_hint or "").strip() if matched else "",
                )
            )
        return roster

    async def _material_from_quote(self, message: Dict[str, Any], stream_id: str) -> str:
        """取被引用消息的正文当素材。

        引用消息在宿主里可能有三种形态，这里依次尝试：

        1. ``raw_message`` 里的 ``{"type": "reply", "data": {...}}`` 段——
           适配器通常会把被引用消息的 ``target_message_content`` 直接塞在这里，
           **这是最常见也最可靠的一路**；
        2. 消息本体的 ``processed_plain_text`` 里带引用的文字；
        3. 顶层 ``reply_to``（仅部分适配器会填）——拿它去 ``message.get_by_id`` 查库。
        """

        if not isinstance(message, dict):
            return ""

        # 路径 1：raw_message 里的 reply 段
        for segment in message.get("raw_message") or []:
            if not isinstance(segment, dict):
                continue
            if str(segment.get("type") or "").lower() != "reply":
                continue
            data = segment.get("data")
            if not isinstance(data, dict):
                continue
            # 宿主序列化用 target_message_content；部分适配器/旧版本用 target_content。
            quoted = str(
                data.get("target_message_content")
                or data.get("target_content")
                or ""
            ).strip()
            if quoted:
                self.ctx.logger.info("从引用段取到素材，长度=%s", len(quoted))
                return quoted
            # 内容为空但有目标 ID，走查库
            target_id = str(data.get("target_message_id") or "").strip()
            if target_id:
                text = await self._lookup_quoted_text(target_id, stream_id)
                if text:
                    return text

        # 路径 2：processed_plain_text 里以 [引用]…[/引用] 包裹的形式带了原消息
        processed = str(message.get("processed_plain_text") or "").strip()
        if processed:
            wrapped = re.search(
                r"\[(?:引用|回复|quoted?|reply)\]\s*(.+?)\s*\[/\s*(?:引用|回复|quoted?|reply)\s*\]",
                processed,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if wrapped:
                quoted = wrapped.group(1).strip()
                if quoted:
                    self.ctx.logger.info("从 [引用]…[/引用] 段取到素材，长度=%s", len(quoted))
                    return quoted

        # 路径 3：顶层 reply_to 查库
        reply_to = str(message.get("reply_to") or "").strip()
        if reply_to:
            text = await self._lookup_quoted_text(reply_to, stream_id)
            if text:
                return text

        self.ctx.logger.warning(
            "没能从引用消息里取到文字：reply_to=%r，raw 段类型=%s",
            message.get("reply_to"),
            [str(s.get("type")) for s in (message.get("raw_message") or []) if isinstance(s, dict)],
        )
        return ""

    async def _lookup_quoted_text(self, message_id: str, stream_id: str) -> str:
        """按被引用消息 ID 查库拿正文。"""

        try:
            record = await self.ctx.message.get_by_id(message_id, chat_id=stream_id)
        except Exception as exc:
            self.ctx.logger.warning("读取引用消息失败：id=%s err=%s", message_id, exc)
            return ""
        text = self._plain_text_of(record)
        if not text:
            self.ctx.logger.warning(
                "引用的消息没有可读文本：id=%s record种类=%s", message_id, type(record).__name__
            )
        return text

    @staticmethod
    def _plain_text_of(record: Any) -> str:
        """从消息字典里拼出纯文本。"""

        if not isinstance(record, dict):
            return ""
        processed = str(record.get("processed_plain_text") or "").strip()
        if processed:
            return processed

        chunks: List[str] = []
        for segment in record.get("raw_message") or []:
            if not isinstance(segment, dict):
                continue
            segment_type = str(segment.get("type") or "")
            data = segment.get("data")
            if segment_type == "text" and isinstance(data, str):
                chunks.append(data)
            elif segment_type == "image":
                chunks.append("[图片]")
            elif segment_type == "at" and isinstance(data, dict):
                chunks.append(f"@{data.get('target_user_nickname') or ''}")
        return "".join(chunks).strip()

    def _platform_allowed_for_stream(self, stream_id: str) -> Optional[bool]:
        """按会话 ID 反查平台并判断是否允许。

        工具与 API 入口拿不到 ``platform``，只能靠命令入口记下的映射。
        查不到该会话的平台时返回 ``None``，表示「无法判断」，由调用方决定如何处理。
        """

        allowed = [str(item).strip().lower() for item in self.config.plugin.only_platforms if str(item).strip()]
        if not allowed:
            # 没配置平台限制，等价于全平台放行。
            return True
        platform = self._stream_platform.get(str(stream_id or "").strip())
        if not platform:
            return None
        return platform in allowed

    def _platform_allowed(self, platform: str) -> bool:
        allowed = [str(item).strip().lower() for item in self.config.plugin.only_platforms if str(item).strip()]
        if not allowed:
            return True
        return str(platform or "").strip().lower() in allowed

    # 模型可能编出来的占位符；这些都不是真实会话 ID。
    _FAKE_STREAM_IDS = frozenset(
        {
            "current", "current_stream", "current_session", "this", "here",
            "now", "default", "unknown", "none", "null", "stream", "chat",
            "当前", "当前会话", "当前聊天", "本会话", "会话", "聊天流",
        }
    )

    def _sanitize_stream_id(self, stream_id: Any) -> str:
        """过滤掉模型编造/占位的 stream_id，只留看起来像真实哈希 ID 的值。

        宿主真实给的会话 ID 是一串十六进制哈希（如 ``c58628a91bab...``），
        而模型调工具时经常自己编一个 ``"current"``，用它发送必然失败。
        """

        value = str(stream_id or "").strip()
        if not value:
            return ""
        if value.lower() in self._FAKE_STREAM_IDS:
            self.ctx.logger.warning("丢弃不可信的 stream_id：%r（疑似模型编造）", value)
            return ""
        # 真实 ID：长度 >= 8 且由字母数字组成
        if len(value) < 8 or not re.fullmatch(r"[0-9a-zA-Z_-]+", value):
            self.ctx.logger.warning("丢弃不像会话 ID 的 stream_id：%r", value)
            return ""
        return value

    def _cooldown_remaining(self, stream_id: str) -> int:
        cooldown = int(self.config.plugin.cooldown_seconds)
        if cooldown <= 0 or not stream_id:
            return 0
        last = self._last_call.get(stream_id)
        if last is None:
            return 0
        remain = cooldown - (time.monotonic() - last)
        return max(0, int(remain + 0.999))

    def _remember(self, stream_id: str, material: str) -> None:
        """记录一次成功出稿，用于冷却与「再来一篇」判断。"""

        if not stream_id:
            return
        self._last_call[stream_id] = time.monotonic()
        self._last_material[stream_id] = material
        self._counter += 1

    @staticmethod
    def _help_text() -> str:
        return (
            "📰 新闻三要素 · 使用说明\n"
            "/新闻三要素 <素材>\n"
            "    默认写固定那三人\n"
            "/新闻三要素 <素材>（人名1，人名2，人名3）\n"
            "    用括号指定死亡名单，多加几个就多死几个\n"
            "/新闻三要素         —— 先引用一条消息再发，用那条消息当素材\n"
            "/新闻三要素 重写    —— 沿用上一份素材再出一版\n"
            "/开头署名 <名称>      —— 设置自己的新闻社（只对自己生效）\n"
            "/开头署名 开 / 关      —— 控制是否加「据○○社报道，」开头\n"
            "/开头署名             —— 查看当前设置\n"
            "/开头署名 重置         —— 恢复为全局默认\n"
            "例：/新闻三要素 美国大选（特朗普，奥巴马，拜登）\n"
            "注：文中人物死亡设定均为虚构玩梗，与现实的任何人和事无关。"
        )


def create_plugin() -> ObituaryNewsroom:
    """插件工厂函数。"""

    return ObituaryNewsroom()
