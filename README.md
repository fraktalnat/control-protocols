# control_protocols

AI control experiments on APPS programming problems, following
[*AI Control: Improving Safety Despite Intentional Subversion*](https://arxiv.org/abs/2312.06942)
(Greenblatt et al., 2023) and ARENA. An untrusted model U writes solutions with subtle backdoors, a
trusted model T rates how suspicious the code looks, and several blue-team protocols
(upfront auditing, trusted monitoring, defer-to-trusted, trusted editing) decide what gets
submitted. To analyze the findings, a Monte-Carlo simulation estimates each protocol's safety.

Tasks run with Inspect.
