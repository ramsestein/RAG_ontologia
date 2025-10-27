# -*- coding: utf-8 -*-
import re

file_path = 'strategies/04_rag_gpt.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Reemplazar emojis con texto
replacements = {
    '\u2705': '[OK]',          # ✅
    '\u26a0\ufe0f': '[WARNING]',  # ⚠️
    '\u26a0': '[WARNING]',      # ⚠
    '\u274c': '[ERROR]',        # ❌
    '\U0001f527': '[FIX]',     # 🔧
    '\U0001f680': '[READY]',   # 🚀
    '\U0001f50d': '[SEARCH]',  # 🔍
    '\U0001f4dd': '[NOTE]',    # 📝
    '\U0001f4cd': '[LOCATION]',# 📍
    '\U0001f50e': '[FIND]',    # 🔎
    '\U0001f3af': '[TARGET]',  # 🎯
    '\U0001f916': '[AI]',      # 🤖
    '\U0001f4e5': '[RESPONSE]',# 📥
    '\u2139\ufe0f': '[INFO]',  # ℹ️
    '\u2139': '[INFO]',         # ℹ
}

for emoji, text in replacements.items():
    content = content.replace(emoji, text)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Emojis removed successfully!')
