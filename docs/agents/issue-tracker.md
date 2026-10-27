# Issue 跟踪器：GitHub

本仓库的 issue 和规格保存在 `Ghost233/GhostSplash` 的 GitHub Issues，使用 `gh` CLI。
所有命令显式指定目标仓库，避免向上游仓库发布。

## 身份与正文

- 写入使用 `Ghost233` 身份，执行前核对账号和目标仓库权限。
- 多账号环境使用单命令或子进程的仓库匹配凭据，保留其他仓库的全局登录状态。
- 密钥仅在进程内传递，不写入文档、命令输出或临时正文。
- 多行正文先保存为 UTF-8 文件，通过 `--body-file` 提交，保留真实换行。

## 常用操作

- 创建：`gh issue create --repo Ghost233/GhostSplash --title "标题" --body-file BODY_FILE`。
- 读取：`gh issue view NUMBER --repo Ghost233/GhostSplash --json number,title,body,labels,comments`。
- 列出：`gh issue list --repo Ghost233/GhostSplash --state open --json number,title,body,labels,comments`，按需增加标签过滤。
- 评论：`gh issue comment NUMBER --repo Ghost233/GhostSplash --body-file BODY_FILE`。
- 标签：`gh issue edit NUMBER --repo Ghost233/GhostSplash --add-label LABEL`；移除使用 `--remove-label LABEL`。
- 关闭：`gh issue close NUMBER --repo Ghost233/GhostSplash`。需要解释时先发布评论。

## Skill 发布约定

skill 要求发布到 issue 跟踪器时，创建 GitHub issue。
`to-spec` 发布完整规格时只加 `ready-for-agent` 标签，不附加其他分类。
发布前检查是否已有同一规格；不因请求超时盲目重复创建。
发布后读取 issue，核对正文、标签、仓库和状态，再报告链接。

## 把 Pull Request 作为分类入口

把 PR 作为请求入口：no。
GitHub issue 与 PR 共用编号空间。编号指向 PR 时，使用对应的 PR 操作，保留上述入口设置。

## 父项与阻塞关系

需要子 issue 时，优先使用当前 GitHub 和 CLI 支持的原生子项关联；不可用时，在子项正文记录父项编号，并在父项维护任务列表。
需要依赖关系时，优先使用原生 issue 依赖；接口使用 issue 的数据库 ID，不能把显示编号作为数据库 ID。
原生依赖不可用时，在正文明确列出阻塞项，查询时核对阻塞项是否仍开放。
