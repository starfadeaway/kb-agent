"""守卫：脚本里的函数调用点必须跟得上函数签名的重构。

## 为什么要有这条

`backend/scripts/smoke.py` 曾经调用
`route_game("绝区零的音擎怎么配", "sr", "sr", "zzz")` —— 4 个位置参数。
而 `route_game` 在一次重构后已经变成 `route_game(question, covered)` 两个参数。
两个后果叠在一起，让这个 bug 藏了很久：

1. 这个调用写在 `if not args.skip_agent:` 分支**里面**。
   带 `--skip-agent` 跑不到它，而 README 第 254 行教 newcomers 跑的
   恰恰是**默认**（不带 flag）那条路径 —— 也就是必然崩的那条。
2. smoke 脚本不在 CI 里跑。它是一个"人肉执行"的自检脚本，
   所以 ruff / pytest / vitest 全绿也照样能烂掉。

**函数改过签名之后，调用点不会自动报错** —— 只有真跑到那一行才会炸。
本守卫把"跑一遍"换成"静态数一遍参数个数"。

## 守卫自身不能假绿

如果哪天 `route_game` 被改名，本文件会一条调用点都找不到，
然后因为"没有不匹配项"而全绿。所以每条断言都要求
**至少命中 N 个已知调用点**，命中数掉到 0 时直接失败。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # 仓库根
BACKEND = ROOT / "backend"

# 一旦有人重命名 route_game 或把它挪进别的模块，请同步改这里 ——
# 改完记得让下面的「扫描确实命中了已知调用点」仍然成立。
GUARDED = {"route_game": 2}

# 每个函数至少应有这么多调用点。少于此说明扫描器本身坏了（改名 / 移文件 / 路径写错）。
MIN_CALL_SITES = {"route_game": 3}


def _python_files() -> list[Path]:
    out: list[Path] = []
    for base in (BACKEND, ROOT):
        for p in base.rglob("*.py"):
            parts = set(p.parts)
            if parts & {".venv", "node_modules", "__pycache__", "build", "dist"}:
                continue
            out.append(p)
    return sorted(set(out))


def _positional_arg_count(node: ast.Call) -> int:
    """位置参数个数。*args 展开的实参无法静态判定，返回 -1 让调用方跳过判定。"""
    if any(isinstance(a, ast.Starred) for a in node.args):
        return -1
    return len(node.args)


def _first_call(src: str) -> ast.Call:
    """从一段源码里取出第一个 Call 节点，供下方的计数逻辑单测使用。"""
    tree = ast.parse(src)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.Call))


@pytest.mark.parametrize(
    "src,expected",
    [
        ("f('a', 'b')", 2),
        ("f('a', covered={})", 2),  # 关键字实参也算数，否则这个合法调用会被误判
        ("f(question='a', covered={})", 2),
        ("f('a')", 1),  # 少传 —— 必须能被判出来
        ("f('a', 'b', 'c')", 3),  # 多传 —— smoke.py 坏的就是这一类
        ("f(*args)", -1),  # 展开调用静态不可判定，必须返回 -1 而不是瞎猜
    ],
)
def test_实参计数把关键字实参也算进去(src: str, expected: int) -> None:
    """计数逻辑本身的单测。

    这里独立于仓库扫描，直接喂源码片段 —— 因为「关键字实参没被算进去」
    这类错误在真实调用点全是位置参数写法的话，永远测不出来。
    """
    node = _first_call(src)
    n_pos = _positional_arg_count(node)
    n_kw = len(node.keywords)
    total = -1 if n_pos == -1 else n_pos + n_kw
    assert total == expected


def _call_sites(func_name: str) -> list[tuple[Path, int, int, int]]:
    """找出所有调用 `func_name` 的位置。

    返回 (文件, 行号, 位置参数个数, 关键字参数个数)。
    只认 `Name` 与 `Attribute` 两种形态（`route_game(...)` / `agent.route_game(...)`），
    认不出 `getattr` / 反射调用 —— 那类不在本守卫覆盖范围内。
    """
    hits: list[tuple[Path, int, int, int]] = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - 仓库内不应出现语法错误文件
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            else:
                continue
            if name == func_name:
                hits.append((path, node.lineno, _positional_arg_count(node), len(node.keywords)))
    return hits


@pytest.mark.parametrize("func_name,expected", sorted(GUARDED.items()))
def test_扫描器本身能命中已知调用点(func_name: str, expected: int) -> None:
    """防假绿的第一道：扫描器必须真的找到调用点。

    没有这一条，`route_game` 被改名后本文件会安静地全绿 ——
    那正是守卫最该避免的失败模式（报告"没问题"而其实根本没看）。
    """
    hits = _call_sites(func_name)
    assert len(hits) >= MIN_CALL_SITES[func_name], (
        f"{func_name} 的调用点只找到 {len(hits)} 个，低于下限 {MIN_CALL_SITES[func_name]}。"
        f"可能发生了重命名 / 移文件 / 路径写错，此时后续断言全部无意义。"
        f"命中：{hits}"
    )


@pytest.mark.parametrize("func_name,expected", sorted(GUARDED.items()))
def test_函数定义的位置参数个数与守卫一致(func_name: str, expected: int) -> None:
    """防假绿的第一道：先确认函数本身就是 N 个位置参数。"""
    react = (BACKEND / "app" / "agent" / "react.py").read_text(encoding="utf-8")
    tree = ast.parse(react)
    defs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == func_name
    ]
    assert len(defs) == 1, f"react.py 里应当恰好定义一个 {func_name}，实际 {len(defs)} 个"
    node = defs[0]
    pos = list(node.args.posonlyargs) + list(node.args.args)
    # 只在首参**确实叫 self/cls** 时才减 —— 无条件减会把模块级函数
    # （route_game 就是）算成少一个参数。
    if pos and pos[0].arg in ("self", "cls"):
        pos = pos[1:]
    assert not node.args.vararg, f"{func_name} 不该有 *args，否则静态判定不可靠"
    n_pos = len(pos)
    assert n_pos == expected, (
        f"{func_name} 定义有 {n_pos} 个位置参数，守卫按 {expected} 个判定。"
        f"两者不一致时调用点断言会给出误导性结果，请同步更新 GUARDED。"
    )


@pytest.mark.parametrize("func_name,expected", sorted(GUARDED.items()))
def test_所有调用点的位置参数个数与定义一致(func_name: str, expected: int) -> None:
    """本体：任何调用点传参个数不对都直接点名文件与行号。"""
    bad = []
    for path, lineno, n_pos, n_kw in _call_sites(func_name):
        if n_pos == -1:  # *args 展开，静态不可判定，跳过（而不是假装它对）
            continue
        # 位置 + 关键字合计才是真正的实参个数。
        # 只数位置参数会把 `route_game("x", covered={})` 误判成"只传了 1 个"。
        if n_pos + n_kw != expected:
            rel = path.relative_to(ROOT)
            bad.append(f"{rel}:{lineno} 传了 {n_pos + n_kw} 个实参（应 {expected}）")
    assert not bad, (
        "以下调用点的位置参数个数与函数定义不符 —— "
        "很可能是函数重构后调用点没跟上（smoke.py 的 route_game 就是这么坏的）：\n  "
        + "\n  ".join(bad)
    )


def test_smoke_py_里的游戏路由检查不在_skip_agent_分支内() -> None:
    """锁定 bug 的第二层成因：不花 token 的检查不能挂在花 token 的分支里。

    smoke.py 的「游戏路由」检查原先位于 `if not args.skip_agent:` 内部。
    带 `--skip-agent` 跑不到，默认跑则必然 TypeError —— 两条路径各错一半，
    没有一条能验证这条检查是活的。路由判定是纯函数，没有理由消耗 token。

    这条断言直接查 AST，不执行脚本（执行要真调 API）。
    """
    src = (BACKEND / "scripts" / "smoke.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    guard_stmts = [n for n in ast.walk(tree) if isinstance(n, ast.If) and n.lineno > 40]
    assert guard_stmts, "smoke.py 结构变了：找不到 --skip-agent 那个 If 节点，请人工确认"

    # 找出每个 route_game 调用点，看它是否落在某个 skip_agent 的 If 体内
    route_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "route_game"
    ]
    assert route_calls, "smoke.py 里没有 route_game 调用点了 —— 若是有意移除，请连同 README 一起改"

    guarded_by_skip = False
    for guard in guard_stmts:
        body_ids = {id(s) for stmt in guard.body for s in ast.walk(stmt)}
        if any(id(c) in body_ids for c in route_calls):
            guarded_by_skip = True

    assert not guarded_by_skip, (
        f"smoke.py:{route_calls[0].lineno} 的 route_game 调用落在 --skip-agent 分支内。"
        "纯函数检查不应挂在消耗 token 的分支下 —— 这样带 flag 跑不到、不带 flag 又必崩。"
    )
