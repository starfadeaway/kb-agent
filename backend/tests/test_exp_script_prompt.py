"""守卫：实验脚本发给模型的系统消息必须是**已格式化**的提示词。

## 缺陷（2026-10-14 读 exp_c.py 时实测发现，非猜测）

`app/agent/react.py` 里的 `SYSTEM_PROMPT` 是**带占位符的模板**：

    SYSTEM_PROMPT = "你是一个游戏知识库助手，知识库覆盖这些游戏：{covered}。
    ...
    各游戏术语不要混用：{terms}
    ...
    {security}"

真正的提示词要过 `build_system_prompt(covered, terms)` 才会把三个占位符填掉。

而 `scripts/exp_c.py` 的无工具单轮档（C0）早先写的是：

    [{"role": "system", "content": SYSTEM_PROMPT}, ...]

——**原样，把模板当消息发了出去**。后果是 C0 档模型实际收到的系统消息里
游戏名是字面量 `{covered}`，而且**注入防护那一整段
`INSTRUCTION_NOTICE` 根本没进 C0 的提示词**（它在模板里就是 `{security}`）。

C1/C2 走 `ReActAgent`，提示词由 `_prepare` 现场 `build_system_prompt` 生成，
**一直是对的** —— 所以这个缺陷只污染 C0，且旧 `exp_C*.json` 里的
C0 与 C1/C2 不可直接比较。

## 为什么这类缺陷能长期存活

因为它**不报错**：HTTP 200、judge 正常打分、分数也在合理区间。
`{covered}` 对模型来说只是一段看不懂的噪声，它照样能回答游戏问题。
所以只能靠断言「发给模型的那段文本里还有没有占位符」来锁住。

## 双向验证不适用

`production_system_prompt` 是本次新增的符号，改造前的 HEAD 上不存在，
没有「修复前快照」可注入。这里用**等价的前置断言**替代：
下方 `test_production_system_prompt_has_no_placeholders` 断言
`production_system_prompt()` 返回值里没有任何 `{...}` 占位符 ——
而改造前 `SYSTEM_PROMPT` 原样发给模型时，这个断言等价成立且必然失败。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
SCRIPTS = BACKEND / "scripts"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(SCRIPTS))

#: `str.format` 用的那套占位符。刻意用 `{security}` / `{covered}` 这种
#: **真实存在过的**字段名，而不是 `{foo}` ——
#: 用假名字的话，模板里任何一处 `{...}` 都会被算成占位符，
#: 断言就退化成「提示词里不能有花括号」，与本缺陷无关。
_PLACEHOLDER = re.compile(r"\{(covered|terms|security)\}")


def test_production_system_prompt_has_no_placeholders():
    """生产提示词里不能残留任何未替换的占位符。

    这是本文件的核心断言。它同时覆盖两个方向：
      - 忘了调用 build_system_prompt（模板被直接发出去）→ 红；
      - format 漏了某个键 → 红。
    """
    from app.agent.react import production_system_prompt

    prompt = production_system_prompt()
    left = _PLACEHOLDER.findall(prompt)
    assert not left, (
        f"生产提示词里还留着未替换的占位符 {sorted(set(left))} —— "
        "有人把未格式化的 SYSTEM_PROMPT 当成提示词发出去了。"
    )
    # 反向确认：注入防护那段必须真的在。
    # 只查「没有占位符」不够 —— 一个空串也能通过那条断言。
    assert "retrieved_data" in prompt, (
        "生产提示词里没有 INSTRUCTION_NOTICE —— "
        "注入防护那段没被 format 进去。"
    )


def test_production_system_prompt_names_the_covered_games():
    """游戏名必须来自 settings.games，而不是占位符。

    早先的硬编码事故是反过来的：提示词写死「星铁与绝区零」，
    加了原神之后模型把原神问题当成越界问题拒答了（EVALUATION.md 有记录）。
    这条锁的是同一个能力的另一个方向。
    """
    from app.agent.react import production_system_prompt
    from app.core.config import get_settings

    prompt = production_system_prompt()
    for slug, name in get_settings().games.items():
        assert name in prompt, f"生产提示词里没有 {slug} 的显示名「{name}」"


@pytest.mark.parametrize("script", ["exp_c.py", "exp_e.py"])
def test_exp_scripts_never_send_the_raw_template(script: str):
    """实验脚本不得把 `SYSTEM_PROMPT` 当系统消息内容发出去。

    这条是**源码级**断言，看起来不如行为断言可靠，但它是唯一能在
    「跑实验之前」就拦住这类错误的东西：等 C0 跑完 15 条才发现提示词是
    模板，API 费用已经花掉了，产物也已经落盘了。

    源码级断言的局限要说明白：它只能挡住「直接用常量名」这一种写法。
    有人写 `SYSTEM_PROMPT.format(...)`（但漏键）、或者自己拼一份提示词常量，
    这条都拦不住 —— 那要靠 4 号测试去发现「成品里还有占位符」。
    两条互补，不能互相替代。
    """
    src = (SCRIPTS / script).read_text(encoding="utf-8")
    bad = re.findall(r'"content":\s*SYSTEM_PROMPT\b', src)
    assert not bad, (
        f"{script} 把未格式化的 SYSTEM_PROMPT 当系统消息内容发出去了。"
        "应改用 production_system_prompt()。"
    )


def test_react_path_and_exp_path_build_the_same_prompt():
    """ReAct 生产路径与实验脚本必须构造**同一份**提示词。

    两边各写一份的话，改了生产忘了改实验（或反过来），
    实验测的就不是线上跑的那个系统了 —— 而且两边各自都自洽，
    没有报错能提示这件事。
    """
    from app.agent.react import ReActAgent, production_system_prompt
    from app.agent.tools import ToolContext
    from app.core.config import get_settings

    s = get_settings()
    ctx = ToolContext(settings=s, retriever=None, store=None, game="sr")
    agent = ReActAgent(settings=s, tools=[])
    _ctx, messages, _state = agent._prepare("三月七的基础生命值是多少", None, None, ctx)
    assert messages[0]["content"] == production_system_prompt()