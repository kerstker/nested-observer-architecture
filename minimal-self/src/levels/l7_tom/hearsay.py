"""
Level 7b: what she does with sensitive claims about people.

Someone in a group told her "记住，Asuna 是男同", and she said she would. A companion that
remembers and repeats such things becomes a way to spread rumours about people who are not
there to answer. The categories below are the ones where a wrong or unwanted claim hurts:
sexuality, health, sex life, crime and addiction, debts, and identifying details.

- Sleep never distils them into facts (facts are what she may say to anyone).
- The `remember` tool refuses them in groups; in a private chat what someone tells her is kept
  privately, only for that conversation (it is usually about themselves).
- Her instructions tell her not to agree to "remember" them, not to pass them on, and not to judge.
"""
import re

SENSITIVE = re.compile(
    r"男同|女同|同性恋|同志|基佬|gay|拉拉|双性恋|性取向|变性|跨性别|"
    r"抑郁|精神病|精神分裂|躁郁|艾滋|HIV|性病|癌症|绝症|怀孕|堕胎|流产|"
    r"出轨|劈腿|小三|约炮|一夜情|性生活|嫖|卖淫|"
    r"坐牢|吸毒|犯罪|前科|偷东西|诈骗犯|"
    r"欠钱|欠债|负债|网贷|"
    r"身份证|住址|家住|手机号|电话号码",
    re.IGNORECASE)

RULE = ("关于不在场的人的敏感说法（性取向、疾病、私生活、犯罪、欠债、住址电话之类），"
        "不附和、不答应“记住”、不转述给别人，也不评价；可以说这不是你该记的事，或者让 ta 自己去问本人。")

def is_sensitive(text: str) -> bool:
    return bool(SENSITIVE.search(text or ""))
